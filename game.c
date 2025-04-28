#include "game.h"
#include "neural_network.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_FG "\033[0m"
#define RED_FG "\033[41m"
#define BLUE_FG "\033[44m"

GameState *game_init() {
  GameState *state = malloc(sizeof(GameState));
  if (!state) {
    perror("game init");
  }

  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      state->board[i * COLS + j] = EMPTY;
    }
  }
  state->turn = BLUE;
  state->winner = NULL;
  return state;
}

int8_t valid_col(GameState *state, uint8_t col) {
  for (uint8_t i = ROWS - 1; i >= 0; i--) {
    if (state->board[i * COLS + col] == EMPTY)
      return i;
  }
  return -1;
}

uint8_t random_move(GameState *state) {
  uint8_t curr = 0;
  uint8_t valid[COLS];
  for (uint8_t i = 0; i < COLS; i++) {
    if (valid_col(state, i)) {
      valid[curr++] = i;
    }
  };
  int col = rand() % curr;
  return valid[col];
}

static void change_turn(GameState *state) {
  if (state->turn == RED) {
    state->turn = BLUE;
  } else {
    state->turn = RED;
  }
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
    if (curr == 4) {
      return &state->turn;
    }
  }
  curr = 0;
  for (char i = 0; i < COLS; i++) {
    continue_check(state->board[row * COLS + i], state->turn, &curr);
    if (curr == 4) {
      return &state->turn;
    }
  }
  curr = 0;
  int8_t srow = row - col;
  int8_t scol = col - row;

  (srow < 0) && (srow = 0);
  (scol < 0) && (scol = 0);
  for (char i = 0; srow + i < ROWS && scol + i < COLS; i++) {
    continue_check(state->board[(srow + i) * COLS + scol + i], state->turn,
                   &curr);
    if (curr == 4) {
      return &state->turn;
    }
  }

  curr = 0;
  srow = row - col;
  scol = col - row;
  (srow < 0) && (srow = 0);
  (scol < 0) && (scol = 0);
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
    printf("%d is not valid\n", col);
    return;
  }

  state->board[row * COLS + col] = state->turn;
  state->winner = check_win(state, row, col);

  if (!state->winner) {
    check_full_board(state);
  }
  if (!state->winner) {
    change_turn(state);
  }
}

void draw_board(GameState *state) {
  char *fg[3] = {DEFAULT_FG, RED_FG, BLUE_FG};
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      printf("%s  ", fg[state->board[i * COLS + j]]); // , state->board[i][j]
    }
    printf(DEFAULT_FG "\n");
  }
  printf(DEFAULT_FG);
  for (char i = 0; i < COLS; i++) {
    printf("%d ", i + 1);
  }
  printf(DEFAULT_FG "\n");
}

uint8_t player_move() {
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
  return col - 1;
}

// uint8_t neural_network_move(GameState *state, NeuralNetwork *nn) { return 0;
// }

BOARD *game_loop(GameState *state, NeuralNetwork *nn) {
  uint8_t col;
  while (state->winner == NULL) {
    if (state->turn == RED) {
      // col = player_move();
      col = random_move(state);
    } else {
      nn_run(nn, (float *)state->board);
      // col = random_move(state);
    }
    insert(state, col);
    // draw_board(state);
  }
  change_turn(state);
  // printf("The winner is ");
  // printf("%s  %s", *state->winner == RED ? RED_FG : BLUE_FG, DEFAULT_FG);
  return state->winner;
}
