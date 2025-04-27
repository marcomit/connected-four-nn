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
      col = max(nn_run(nn, board(state)), nn->layers[nn->size - 1]->size);
      if (valid_col(state, col) == -1) {
        nn_train(nn, -1.0f);
      }
    } else {
      col = random_move(state);
    }
    insert(state, col);
  }
  float reward = 1.0f;
  if (state->turn == RED) {
    reward = -1.0f;
  }
  nn_train(nn, reward);
}

int main(int argc, char **argv) {
  srand(time(NULL));

  size_t games = 10000;
  // Creation of the network struct
  NeuralNetwork *nn = nn_create(5, 0.01);
  // Defining the layers
  nn->layers[0] = dense(ROWS * COLS, RELU, NULL);
  nn->layers[1] = dense(128, RELU, NULL);
  nn->layers[2] = dense(64, RELU, NULL);
  nn->layers[3] = dense(32, RELU, NULL);
  nn->layers[4] = dense(COLS, NULL, softmax);
  // Initialize the weights (they must be initialized after the definition of
  // the layers)
  nn_init_weights(nn);

  for (int i = 0; i < games; i++) {
    play_game(nn);
  }

  return 0;
}
