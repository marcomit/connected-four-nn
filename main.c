#include "ann.h"
#include "game.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BOARD_LEN (ROWS) * (COLS)

typedef NeuralNetworkEntry NNE;
typedef uint8_t (*GMF)(GameState *, NeuralNetwork *);

float *normalize_board(GameState *game) {
  float *res = malloc(sizeof(float) * BOARD_LEN);
  for (int i = 0; i < BOARD_LEN; i++) {
    if (game->board[i] == EMPTY) {
      res[i] = 0;
    } else if (game->board[i] == game->turn) {
      res[i] = 1;
    } else {
      res[i] = 2;
    }
  }
  return res;
}
void train(NeuralNetwork *net, float reward, GameState *game) {
  float *target = malloc(sizeof(float) * COLS);
  NeuralNetworkLayer *out = net->layers[net->len - 1];
  size_t output_size = out->len;

  for (size_t i = 0; i < output_size; i++) {
    target[i] = 0.1f;
  }

  float discount_factor = 0.9f; // 0.99f;
  float current_reward = reward;

  for (size_t i = 0; i < game->move_count - 1; i++) {
    float *out = nnforward(net, game->history[i + 1].inputs);
    size_t len = net->layers[net->len - 1]->len;
    current_reward = nnmax(out, len);
    target[game->history[i].taken] = reward + discount_factor * current_reward;

    // target[game->history[i].taken] = current_reward;
    // current_reward *= discount_factor;
  }

  nnbalance(net, target);
}

int8_t take_valid_col(GameState *state, float *out) {
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

uint8_t net_move(GameState *game, NeuralNetwork *net) {

  nnforesee(net, normalize_board(game));

  int8_t generated = take_valid_col(game, net->layers[net->len - 1]->a);
  if (generated == -1) {
    printf("Rete rotta");
  } else {
    return generated;
  }
  printf("E' finito proprio dove non doveva finire\n\n\n");
  exit(1);
}

GameState *gameplay(NeuralNetwork *net, GMF blue, GMF red, int show) {
  GameState *game = game_init();
  while (!game->winner) {
    uint8_t col;
    if (game->turn == RED) {
      col = red(game, net);
    } else {
      col = blue(game, net);
    }
    insert(game, col);
    if (show)
      draw_board(game);
  }
  return game;
}

int8_t game(NeuralNetwork *net, GMF func) {
  GMF enemy = func;
  GMF net_func = net_move;
  BOARD net_player = RED;
  // if (rand() % 2)
  if (0) {
    enemy = net_move;
    net_func = func;
    net_player = BLUE;
  }
  GameState *g = gameplay(net, enemy, net_func, 0);

  float reward = -1.0f;
  if (*g->winner == net_player) {
    reward = 1.0f;
  } else if (*g->winner == EMPTY) {
    reward = 0.0f;
  }

  train(net, reward, g);
  if (*g->winner == EMPTY)
    return 0;
  return *g->winner == net_player ? 1 : -1;
}

void print_stats(size_t w, size_t l, size_t p, size_t t) {
  printf("TOTAL: %zu WINS: %zu LOSES: %zu DRAWS: %zu\n", t, w, l, p);
}

int main(int argc, char **argv) {
  srand(time(NULL));
  int games = 10000;

  NeuralNetwork *net = nncreate(5, 0.1f);
  // net->layers[0] = dense(42, RELU);
  // net->layers[1] = dense(128, RELU);
  // net->layers[2] = dense(128, RELU);
  // net->layers[3] = dense(64, RELU);
  // net->layers[4] = dense(7, SOFTMAX);
  //
  // nninit(net);

  nnload(net, "rl");

  size_t twins = 0;
  size_t tloses = 0;
  size_t tdraws = 0;

  size_t wins = 0;
  size_t loses = 0;
  size_t draws = 0;

  GMF func[2] = {random_move, net_move};
  for (int f = 0; f < 2; f++)
    for (int i = 1; i <= games; i++) {
      int result = game(net, func[0]);
      if (result == 0) {
        draws++;
      } else if (result == 1) {
        wins++;
      } else
        loses++;
      if (i % 100 == 0) {
        twins += wins;
        tloses += loses;
        tdraws += draws;
        print_stats(wins, loses, draws, twins + tloses + tdraws);
        wins = 0;
        loses = 0;
        draws = 0;
      }
      // free_game(g);
    }

  printf("\n");
  print_stats(twins, tloses, tdraws, twins + tloses + tdraws);

  printf("\n");
  printf("WINS: %.2f\n",
         (float)(100 * twins) / (float)(twins + tloses + tdraws));

  nnsave(net, "rl");

  GameState *g = gameplay(net, player_move, net_move, 1);

  return 0;
}
