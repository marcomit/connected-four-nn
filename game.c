#include "game.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_FG "\033[0m"
#define RED_FG "\033[41m"
#define BLUE_FG "\033[44m"

static int8_t max(int8_t a, int8_t b) { return a > b ? a : b; }

void free_game(GameState *g) {
  for (size_t i = 0; i < g->move_count; i++)
    free(g->history[i].inputs);

  free(g->winner);
  free(g->history);
  free(g);
}

GameState *game_init() {
  GameState *state = malloc(sizeof(GameState));
  if (!state) {
    perror("game init");
  }

  for (uint8_t i = 0; i < ROWS; i++)
    for (uint8_t j = 0; j < COLS; j++)
      state->board[i * COLS + j] = EMPTY;

  state->turn = RED;
  state->winner = NULL;

  state->history = malloc(sizeof(GameMove) * ROWS * COLS);

  return state;
}

int8_t valid_col(GameState *state, uint8_t col) {
  for (uint8_t i = ROWS - 1; i >= 0; i--)
    if (state->board[i * COLS + col] == EMPTY)
      return i;
  return -1;
}

uint8_t random_move(GameState *state, NeuralNetwork *net, BOARD t) {
  uint8_t curr = 0;
  uint8_t valid[COLS];
  for (uint8_t i = 0; i < COLS; i++)
    if (valid_col(state, i))
      valid[curr++] = i;
  int col = rand() % curr;
  return valid[col];
}

static inline void continue_check(BOARD cell, BOARD turn, uint8_t *curr) {
  if (cell != turn) {
    *curr = 0;
    return;
  }
  (*curr)++;
}

static BOARD *check_win(GameState *state, uint8_t row, uint8_t col) {
  uint8_t curr = 0;
  for (char i = 0; i < ROWS; i++) {
    continue_check(state->board[i * COLS + col], state->turn, &curr);
    if (curr == 4)
      return &state->turn;
  }
  curr = 0;
  for (char i = 0; i < COLS; i++) {
    continue_check(state->board[row * COLS + i], state->turn, &curr);
    if (curr == 4)
      return &state->turn;
  }
  curr = 0;
  int8_t srow = max(row - col, 0);
  int8_t scol = max(col - row, 0);

  for (char i = 0; srow + i < ROWS && scol + i < COLS; i++) {
    continue_check(state->board[(srow + i) * COLS + scol + i], state->turn,
                   &curr);
    if (curr == 4) {
      return &state->turn;
    }
  }

  curr = 0;
  srow = max(row - col, 0);
  scol = max(col - row, 0);
  for (char i = 0; srow + i < ROWS && scol + i < COLS; i++) {
    continue_check(state->board[(srow + i) * COLS + scol + i], state->turn,
                   &curr);
    if (curr == 4) {
      return &state->turn;
    }
  }
  return NULL;
}

static void check_full_board(GameState *state) {
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      if (state->board[i * COLS + j] == EMPTY)
        return;
    }
  }
  *state->winner = EMPTY;
}
void insert(GameState *state, uint8_t col) {
  int8_t row = valid_col(state, col);
  if (row == -1) {
    return;
  }

  state->board[row * COLS + col] = state->turn;
  state->winner = check_win(state, row, col);

  GameMove move;

  move.inputs = malloc(sizeof(float) * ROWS * COLS);

  for (size_t i = 0; i < (size_t)(ROWS * COLS); i++) {
    move.inputs[i] = (float)state->board[i];
  }
  move.taken = col;

  state->history[state->move_count++] = move;

  if (!state->winner) {
    check_full_board(state);
  }
  if (!state->winner) {
    state->turn = state->turn == RED ? BLUE : RED;
  }
}

void draw_board(GameState *state) {
  char *fg[3] = {DEFAULT_FG, RED_FG, BLUE_FG};
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      printf("%s  ", fg[state->board[i * COLS + j]]);
    }
    printf(DEFAULT_FG "\n");
  }
  printf(DEFAULT_FG);
  for (char i = 0; i < COLS; i++) {
    printf("%d ", i + 1);
  }
  printf(DEFAULT_FG "\n");
}

uint8_t player_move(GameState *state, NeuralNetwork *net, BOARD t) {
  int col = -1;
  while (col == -1) {
    printf("Inserisci la colonna: ");
    if (scanf("%d", &col) == -1) {
      col = -1;
    }
    if (col < 1 || col > COLS) {
      col = -1;
    }
    if (col == -1) {
      printf("Numero inserito non valido\n");
    }
  }
  if (valid_col(state, col - 1) == -1)
    return player_move(state, net, t);
  return col - 1;
}
