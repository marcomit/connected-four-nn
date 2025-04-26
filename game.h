#ifndef GAME_H
#define GAME_H

#include <stdint.h>

#define ROWS 6
#define COLS 7

typedef enum { EMPTY, RED, BLUE } BOARD;

// Forward declaration of NeuralNetwork
typedef struct NeuralNetwork NeuralNetwork;

typedef struct {
  BOARD board[ROWS * COLS];
  BOARD *winner;
  BOARD turn;
} GameState;

GameState *game_init();
uint8_t random_col(GameState *state);
void insert(GameState *state, uint8_t col);
void draw_board(GameState *state);

BOARD *game_loop(GameState *state, NeuralNetwork *nn);

#endif // GAME_H
