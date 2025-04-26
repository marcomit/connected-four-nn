// #include "game.h"
#include "neural_network.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  printf("MAIN------------");
  srand(time(NULL));
  nn_train(10);

  // GameState *state = game_init();

  // for (int i = 0; i < argc; i++) {
  //   printf("%s\n", argv[i]);
  // }
  return 0;
}
