#include <stddef.h>
#include <stdint.h>
#ifndef ANN_H
#define ANN_H

typedef enum {
  // Ridge functions
  IDENTITY,
  RELU,
  SIGMOID,
  SWISH,
  TANH,
  SOFTSIGN,

  SOFTPLUS,
  ELISH,
  SINUSOID,
  GAUSSIAN,

  SOFTMAX,
  MAXOUT
  // Radial functions
} NeuralNetworkActivation;

typedef NeuralNetworkActivation NNA;
typedef struct NeuralNetworkEntry NeuralNetworkEntry;

typedef struct {
  // Pre activation
  float *z;

  // Post activation
  float *a;

  // Matrice dei pesi
  float **W;

  // Biases
  float *b;

  // Gradienti dei neuroni
  float *d;

  // Gradienti dei biases
  float *db;

  // Gradienti dei pesi
  float **dW;

  NNA activation;

  size_t len;
} NeuralNetworkLayer;

struct NeuralNetworkEntry {
  float *inputs;
  uint16_t taken;
  NeuralNetworkEntry *next;
};

typedef struct {
  float learning_rate;
  NeuralNetworkLayer **layers;
  NeuralNetworkEntry *history;
  size_t len;
} NeuralNetwork;

NeuralNetwork *nncreate(size_t, float);
void nninit(NeuralNetwork *);

NeuralNetworkLayer *dense(size_t, NeuralNetworkActivation);

void nnforesee(NeuralNetwork *, float *);
void nnbalance(NeuralNetwork *, float *);

void nnstoreentry(NeuralNetwork *, uint16_t taken);
void nnfreehistory(NeuralNetworkEntry *);

#endif /* ifndef ANN_H */
