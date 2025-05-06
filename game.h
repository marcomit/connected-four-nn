#ifndef GAME_H
#define GAME_H
#include "ann.h"
#include <stddef.h>
#include <stdint.h>

#define ROWS 6
#define COLS 7

typedef enum { EMPTY, RED, BLUE } BOARD;

typedef struct GameMove {
  float *inputs;
  uint8_t taken;
} GameMove;

typedef struct {
  BOARD board[ROWS * COLS];
  BOARD *winner;
  BOARD turn;

  GameMove *history;
  size_t move_count;
} GameState;

GameState *game_init();

uint8_t random_move(GameState *, NeuralNetwork *);
uint8_t player_move(GameState *, NeuralNetwork *);

int8_t valid_col(GameState *, uint8_t);
void insert(GameState *, uint8_t);
void draw_board(GameState *);
void free_game(GameState *);
#endif // GAME_H
