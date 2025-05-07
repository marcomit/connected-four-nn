#include "nnlayer.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifndef ANN_H
#define ANN_H

typedef struct {
  float learning_rate;
  NeuralNetworkLayer **layers;
  size_t len;
} NeuralNetwork;

NeuralNetwork *nncreate(size_t, float);
void nninit(NeuralNetwork *);

NeuralNetworkLayer *nndense(size_t, NeuralNetworkActivation);

void nnforesee(NeuralNetwork *, float *);
float *nnforward(NeuralNetwork *, float *);
void nnbalance(NeuralNetwork *, float *);

float nnmax(float *, size_t);

void nnsave(NeuralNetwork *, const char *);
bool nnload(NeuralNetwork *, const char *);
#endif /* ifndef ANN_H */
