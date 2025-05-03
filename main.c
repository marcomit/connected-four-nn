#include "ann.h"
#include "game.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef NeuralNetworkEntry NNE;

float *from_board(BOARD *b, size_t len) {
  float *in = malloc(sizeof(float) * len);
  for (size_t i = 0; i < len; i++) {
    in[i] = (float)(b[i]);
  }
  return in;
}

BOARD *to_board(float *in, size_t len) {
  BOARD *b = malloc(sizeof(BOARD) * len);
  for (size_t i = 0; i < len; i++) {
    b[i] = (BOARD)(in[i]);
  }
  return b;
}

void train(NeuralNetwork *net, float reward) {
  float *target = malloc(sizeof(float) * COLS);
  NeuralNetworkLayer *out = net->layers[net->len - 1];
  size_t output_size = out->len;

  for (size_t i = 0; i < output_size; i++) {
    target[i] = 0.1f;
  }
  size_t move_count = 0;
  NeuralNetworkEntry *current = net->history;
  while (current != NULL) {
    move_count++;
    current = current->next;
  }

  float discount_factor = 0.9f;
  float current_reward = reward;

  NeuralNetworkEntry **moves =
      (NeuralNetworkEntry **)malloc(move_count * sizeof(NeuralNetworkEntry *));
  if (!moves)
    return;
  size_t idx = move_count - 1;
  current = net->history;
  while (current != NULL) {
    moves[idx--] = current;
    current = current->next;
  }

  for (size_t i = 0; i < move_count; i++) {
    target[moves[i]->taken] = current_reward;

    current_reward *= discount_factor;
  }

  nnbalance(net, target);

  free(moves);
}
int8_t take_valid_col(GameState *state, float *out) {
  float epsilon = 0.2f;

  if ((float)rand() / RAND_MAX < epsilon) {
    int valid_cols[COLS];
    int valid_count = 0;

    for (int i = 0; i < COLS; i++) {
      if (valid_col(state, i) != -1) {
        valid_cols[valid_count++] = i;
      }
    }

    if (valid_count > 0) {
      return valid_cols[rand() % valid_count];
    }
  }

  int8_t m = -1;
  for (int i = 0; i < COLS; i++) {
    if (valid_col(state, i) == -1) {
      continue;
    }
    if (m == -1) {
      m = i;
    }
    if (out[i] > out[m])
      m = i;
  }
  return m;
}
BOARD *gameplay(NeuralNetwork *net, uint8_t (*move)(GameState *), int show) {
  size_t board_len = ROWS * COLS;
  GameState *game = game_init();
  while (!game->winner) {
    uint8_t col;
    if (game->turn == RED) {
      nnforesee(net, from_board(game->board, board_len));

      int8_t generated = take_valid_col(game, net->layers[net->len - 1]->a);
      if (generated == -1) {
        printf("Rete rotta");
      } else {
        nnstoreentry(net, generated);
        for (int i = 0; i < COLS; i++) {
          // printf("%2f ", net->layers[net->len - 1]->a[i]);
        }
        // printf("choosen %d\n", generated);
        col = generated;
      }
    } else {
      col = move(game);
    }
    insert(game, col);
    if (show)
      draw_board(game);
    // getchar();
  }
  // printf("---------------------------------------------\n");
  // printf("---------------------------------------------\n");
  // printf("---------------------------------------------\n");
  return game->winner;
}

BOARD *game(NeuralNetwork *net) {
  BOARD *winner = gameplay(net, random_move, 0);
  // printf("The winner is %s\n", *game->winner == RED ? "red" : "blue");

  // calcolare i target
  float reward = 0.0f;
  if (*winner == RED) {
    reward = 1.0f;
  } else if (*winner == BLUE) {
    reward = -0.5f;
  }

  train(net, reward);
  nnfreehistory(net->history);
  return winner;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  int games = 10000;
  NeuralNetwork *net = nncreate(5, 0.1f);
  net->layers[0] = dense(42, RELU);
  net->layers[1] = dense(128, RELU);
  net->layers[2] = dense(64, RELU);
  net->layers[3] = dense(32, RELU);
  net->layers[4] = dense(7, SOFTMAX);
  nninit(net);

  size_t wins = 0;
  size_t loses = 0;
  size_t pairs = 0;

  for (int i = 0; i < games; i++) {
    BOARD *winner = game(net);
    if (*winner == EMPTY) {
      pairs++;
    } else if (*winner == RED) {
      wins++;
    } else
      loses++;
  }
  printf("Total: %zu\n", loses + wins + pairs);
  printf("%zu wins\n", wins);
  printf("%zu pairs\n", pairs);
  printf("%zu loses\n", loses);

  gameplay(net, player_move, 1);
  return 0;
}
