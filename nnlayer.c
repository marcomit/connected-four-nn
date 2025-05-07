#include "nnlayer.h"
#include <stdio.h>
#include <stdlib.h>

typedef NeuralNetworkDenseParams NNDense;
typedef NeuralNetworkConv2DParams CNN;
typedef NeuralNetworkLSTMParams LSTM;

static void dense_forward_pass(NNDense *curr, float *input, size_t len) {

  for (size_t i = 0; i < curr->len; i++) {
    float z = curr->b[i];
    for (size_t j = 0; j < len; j++) {
      z += input[j] * curr->W[i][j];
    }
    curr->z[i] += z;
    curr->a[i] += z;
  }
  perror("It must be activated");
  // activation(curr);
}
static void dense_backpropagation(NNDense *l, float *input, size_t len) {}
static void dense_free(NNDense *params) {
  free(params->d);
  free(params->db);
  free(params->a);
  free(params->b);
  free(params->z);
  for (size_t i = 0; i < params->len; i++) {
    free(params->W[i]);
    free(params->dW[i]);
  }
  free(params);
}
// static void dense() {}
NNL *dense(size_t len, NNA activation) {
  NNL *layer = malloc(sizeof(NeuralNetworkLayer));

  layer->type = LAYER_DENSE;

  NNDense *params = malloc(sizeof(NNDense));

  params->activation = activation;
  params->len = len;
  params->a = calloc(len, sizeof(float));
  params->z = calloc(len, sizeof(float));
  params->b = calloc(len, sizeof(float));
  params->db = calloc(len, sizeof(float));
  params->d = calloc(len, sizeof(float));

  layer->params = params;

  layer->backward = (void (*)(void *, float *, size_t))dense_backpropagation;
  layer->forward = (void (*)(void *, float *, size_t))dense_forward_pass;

  return layer;
}

static void conv2d_forward_pass(NNL *l, float *input, size_t len) {}
static void conv2d_backpropagation(NNL *l, float *input, size_t len) {}
static void conv2d_free(CNN *cnn) {
  free(cnn->b);
  free(cnn->db);
  free(cnn->col_buffer);
  free(cnn->d_kernels);
  free(cnn->kernels);
  free(cnn->input_cache);
  free(cnn);
}
NNL *conv2d() {
  NNL *layer = malloc(sizeof(NNL));

  layer->type = LAYER_CNN;

  CNN *cnn = malloc(sizeof(CNN));

  // cnn->input_cache = calloc(len, sizeof(float));

  layer->params = cnn;
  return layer;
}
static void lstm_forward_pass(LSTM *l, float *input, size_t len) {}
static void lstm_backpropagation(LSTM *l, float *input, size_t len) {}
static void lstm_free(LSTM *l) { free(l); }
NNL *lstm() {
  NNL *l = malloc(sizeof(NNL));
  LSTM *params = malloc(sizeof(LSTM));
  l->params = params;
  return l;
}
