// #include "game.h"
#include "game.h"
#include "neural_network.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

uint8_t max(float *inputs, size_t size) {
  uint8_t res = 0;
  for (size_t i = 0; i < size; i++) {
    inputs[res] < inputs[i] && (res = i);
  }
  return res;
}

float *board(GameState *state) {
  float *inputs = malloc(sizeof(float) * ROWS * COLS);
  for (int i = 0; i < ROWS * COLS; i++) {
    inputs[i] = (float)(state->board[i]);
  }
  return inputs;
}

void play_game(NeuralNetwork *nn) {
  GameState *state = game_init();
  while (!state->winner) {
    uint8_t col;
    if (state->turn == RED) {
      NNLayer *out = nn->layers[nn->size - 1];
      nn_run(nn, board(state));
      col = max(out->outputs, out->size);
      if (valid_col(state, col) == -1) {
        getchar();
        // break;
        printf("Ha fatto una mossa sbagliata %d\n", col);
        nn_train(nn, -0.001f);
      } else {
        insert(state, col);
      }
    } else {
      col = random_move(state);
      insert(state, col);
    }
  }
  printf("Ha vinto %s\n", *state->winner == RED ? "AI" : "BLUE");
  // getchar();
  float reward = 1.0f;
  if (*state->winner == BLUE) {
    reward = -1.0f;
  }
  nn_train(nn, reward * 0.001f);
  nn_free_history(nn);
}

int main() {
  srand(time(NULL));
  size_t games = 10;
  // Creation of the network struct
  NeuralNetwork *nn = nn_create(5, 0.01, ROWS * COLS);
  // Defining the layers
  nn->layers[0] = dense(ROWS * COLS, RELU, NULL);
  nn->layers[1] = dense(128, RELU, NULL);
  nn->layers[2] = dense(64, RELU, NULL);
  nn->layers[3] = dense(32, RELU, NULL);
  nn->layers[4] = dense(COLS, FLAT, softmax);
  // Initialize the weights (they must be initialized after the definition of
  // the layers)
  nn_init(nn);

  for (size_t i = 0; i < games; i++) {
    play_game(nn);
  }

  return 0;
}
