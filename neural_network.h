#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "game.h"
#include <stddef.h>
#include <stdint.h>

typedef struct NNLayer NNLayer;

typedef enum { RELU, LEAKY_RELU, ELU, SIGMOID, SWISH, FLAT } Activation;

struct NNLayer {
  size_t size;

  float *biases;

  // Sono i neuroni del layer
  float *outputs;

  // Gradienti.
  // Servono a capire come deve cambiare il peso.
  // Cioe' dice al peso se andare su o giu per migliorare
  float *deltas;

  float **weights;

  // Funzione di attivazione
  float (*activation)(float);

  // Funzione che normalizza l'outputs
  // Viene usata solo per l'ultimo layer dove viene applicato softmax.
  void (*normalize)(NNLayer *);
};

typedef struct NeuralNetwork {
  NNLayer **layers;
  size_t size;
  float learning_rate;
} NeuralNetwork;

NeuralNetwork *nn_create(size_t, float);
uint8_t nn_run(NeuralNetwork *, GameState *);
void nn_train(uint16_t);

#endif // NEURAL_NETWORK_H
