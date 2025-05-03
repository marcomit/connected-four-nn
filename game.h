#ifndef GAME_H
#define GAME_H
#include "ann.h"
#include <stdint.h>

#define ROWS 6
#define COLS 7

typedef enum { EMPTY, RED, BLUE } BOARD;

typedef struct {
  BOARD board[ROWS * COLS];
  BOARD *winner;
  BOARD turn;
} GameState;

GameState *game_init();

uint8_t random_move(GameState *);
uint8_t player_move(GameState *);

int8_t valid_col(GameState *, uint8_t);
void insert(GameState *, uint8_t);
void draw_board(GameState *);

#endif // GAME_H
