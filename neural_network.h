#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "game.h"
#include <stddef.h>
#include <stdint.h>

typedef struct NNLayer NNLayer;
typedef struct NNHistory NNHistory;

typedef enum { RELU, LEAKY_RELU, ELU, SIGMOID, SWISH, FLAT } Activation;

struct NNLayer {
  size_t size;

  // Bias (Neuroni prima dell'attivazione)
  float *biases;

  // Sono i neuroni del layer
  float *outputs;

  // Pesi
  float **weights;

  // Funzione di attivazione
  Activation acttype;

  // Funzione che normalizza l'outputs
  // Viene usata solo per l'ultimo layer dove viene applicato softmax.
  void (*normalize)(NNLayer *);

  // Gradienti.
  // Servono a capire come deve cambiare il peso.
  // Cioe' dice al peso se andare su o giu per migliorare
  float *deltas;
  float **deltas_weights;
  float *deltas_biases;
};

struct NNHistory {
  float *input;
  uint8_t taken;
};

typedef struct NeuralNetwork {
  NNLayer **layers;
  size_t size;
  float learning_rate;

  NNHistory *history;
  uint8_t num_moves;
} NeuralNetwork;

void softmax(NNLayer *);

NeuralNetwork *nn_create(size_t, float, size_t);

NNLayer *dense(size_t, Activation, void (*normalize)(NNLayer *));

void nn_init(NeuralNetwork *);
void nn_run(NeuralNetwork *, float *);
void nn_save_move(NeuralNetwork *);
void nn_train(NeuralNetwork *, float);
void nn_free_history(NeuralNetwork *);
#endif // NEURAL_NETWORK_H
