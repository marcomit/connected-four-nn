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

uint8_t random_move(GameState *);
int8_t valid_col(GameState *, uint8_t);
void insert(GameState *, uint8_t);
void draw_board(GameState *);

BOARD *game_loop(GameState *, NeuralNetwork *);

#endif // GAME_H
