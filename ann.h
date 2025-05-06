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

typedef struct {
  float learning_rate;
  NeuralNetworkLayer **layers;
  size_t len;
} NeuralNetwork;

NeuralNetwork *nncreate(size_t, float);
void nninit(NeuralNetwork *);

NeuralNetworkLayer *dense(size_t, NeuralNetworkActivation);

void nnforesee(NeuralNetwork *, float *);
float *nnforward(NeuralNetwork *, float *);
void nnbalance(NeuralNetwork *, float *);

float nnmax(float *, size_t);

void nnsave(NeuralNetwork *, const char *);
void nnload(NeuralNetwork *, const char *);
#endif /* ifndef ANN_H */
