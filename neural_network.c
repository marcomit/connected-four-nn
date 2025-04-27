#include "neural_network.h"
#include "game.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

// Funzione per generare numeri casuali compresi tra -x e +x
#define RANDOM(x) (float)rand() / RAND_MAX *(float)(x) * 2 - (float)(x)

static float sum(float *inputs, size_t size) {
  float res = 0;
  for (size_t i = 0; i < size; i++) {
    res += inputs[i];
  }
  return res;
}

static uint8_t max(float *inputs, size_t size) {
  uint8_t res = 0;
  for (size_t i = 0; i < size; i++) {
    inputs[res] < inputs[i] && (res = i);
  }
  return res;
}

// Activation functions
static float relu(float x) { return x > 0 ? x : 0; }
static float leaky_relu(float x) { return x > 0 ? x : 0.01f * x; }
static float elu(float x) { return x > 0 ? x : 1 * expf(-x); }
static float sigmoid(float x) { return 1.0f / (float)(1.0f + expf(-x)); }
static float swish(float x) { return x * sigmoid(x); }

// Derivative activation functions
static float drelu(float x) { return x > 0 ? 1.0f : 0; }
static float dleaky_relu(float x) { return x > 0 ? 1 : 0.01f; }
static float delu(float x) { return x > 0 ? 1 : -expf(-x); }
static float dsigmoid(float x) { return logf(1.0f + 1); } // TODO
static float dswish(float x) { return x; }                // TODO

void softmax(NNLayer *layer) {
  float *inputs = layer->outputs;
  int size = layer->size;
  float s = sum(inputs, size);
  // float m = inputs[max(inputs, size)];
  if (s > 0) {
    for (int i = 0; i < size; i++) {
      inputs[i] /= s;
    }
  }
  for (int i = 0; i < size; i++) {
    inputs[i] = 1.0f / size;
  }
}

float activation(Activation act, float in) {
  float (*func[6])(float) = {relu, leaky_relu, elu, sigmoid, swish, NULL};
  return func[act](in);
}

float derivate_activation(Activation act, float in) {
  float (*func[6])(float) = {drelu, dleaky_relu, delu, dsigmoid, dswish, NULL};
  return func[act](in);
}

NNLayer *dense(size_t size, Activation acttype, void (*normalize)(NNLayer *)) {
  NNLayer *l = malloc(sizeof(NNLayer));
  l->acttype = acttype;
  l->normalize = normalize;
  l->size = size;
  l->weights = NULL;

  l->biases = (float *)calloc(l->size, sizeof(float));
  l->outputs = (float *)calloc(l->size, sizeof(float));
  l->deltas = (float *)calloc(l->size, sizeof(float));
  l->deltas_biases = (float *)calloc(l->size, sizeof(float));

  return l;
}

static void forward_pass_layer(NeuralNetwork *nn, int index) {
  NNLayer *prev = nn->layers[index];
  NNLayer *next = nn->layers[index + 1];
  float **w = prev->weights;
  for (size_t i = 0; i < next->size; i++) {
    float s = 0;
    for (size_t j = 0; j < prev->size; j++) {
      s += prev->biases[j] * w[j][i];
    }
    s += next->biases[i];
    float output = s;
    if (prev->acttype != FLAT) {
      output = activation(prev->acttype, s);
    }
    next->outputs[i] = output;

    if (next->normalize) {
      next->normalize(next);
    }
  }
}

static void forward_pass(NeuralNetwork *nn, float *inputs) {
  memcpy(nn->layers[0]->biases, inputs, nn->layers[0]->size);
  for (int i = nn->size - 1; i > 1; i--) {
    forward_pass_layer(nn, i - 1);
  }
}

void nn_init(NeuralNetwork *nn) {
  // Set the weights for each layer
  for (size_t i = 0; i < nn->size - 1; i++) {
    NNLayer *curr = nn->layers[i];
    NNLayer *next = nn->layers[i + 1];

    curr->weights = (float **)malloc(sizeof(float *) * curr->size);
    curr->deltas_weights = (float **)malloc(sizeof(float *) * curr->size);

    for (size_t j = 0; j < curr->size; j++) {
      curr->weights[j] = (float *)malloc(sizeof(float) * next->size);
      curr->deltas_weights[j] = (float(*))malloc(sizeof(float) * next->size);
    }
  }
}

static void free_history(NNHistory *curr) {
  if (!curr) {
    return;
  }
  free(curr->input);
  free_history(curr->next);
}

static void add_history(NeuralNetwork *nn) {
  NNHistory *history = malloc(sizeof(NNHistory));
  NNLayer *in = nn->layers[0];
  NNLayer *out = nn->layers[nn->size - 1];

  history->input = malloc(sizeof(float) * in->size);
  memcpy(in->outputs, history->input, in->size);
  history->taken = max(out->outputs, out->size);

  // history->next = malloc(sizeof(NNHistory));
  history->next = nn->history;
  nn->history = history;
  // return history;
}

float *nn_run(NeuralNetwork *nn, float *inputs) {
  forward_pass(nn, inputs);
  NNLayer *output = nn->layers[nn->size - 1];
  for (int i = 0; i < COLS; i++) {
    printf("%d, ", (int)(output->outputs[i] * 100));
  }
  printf("\n");
  add_history(nn);
  return output->outputs;
}

NeuralNetwork *nn_create(size_t size, float learning_rate) {
  printf("creazione rete\n");
  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->layers = (NNLayer **)malloc(sizeof(NNLayer *) * size);
  nn->size = size;
  nn->learning_rate = learning_rate;

  return nn;
}

static void update_deltas(NeuralNetwork *nn, float reward) {
  NNLayer *out = nn->layers[nn->size - 1];
  uint8_t m = max(out->outputs, out->size);
  for (size_t i = 0; i < out->size; i++) {
    int s = i == m ? 1 : 0;
    out->deltas[i] = (s - out->outputs[i]) * reward;
  }

  for (int i = nn->size - 2; i > 0; i--) {
    NNLayer *curr = nn->layers[i];
    derivate_activation(curr->acttype, curr->outputs[i]);
  }
  // Devo aggiornare i gradienti del layer in base al reward.
  // Per l'azione scelta (nel mio caso con la probabilita piu alta)
  // il gradiente sara' deltas[azione_scelta] = (1 - probabilita_scelta) *
  // reward
  // Per le altre azioni il gradiente sara (-probabilita) * reward
}

static void update_weights(NeuralNetwork *nn) {
  for (int i = 0; i < nn->size - 1; i++) {
    NNLayer *curr = nn->layers[i];
    NNLayer *next = nn->layers[i + 1];
    for (int j = 0; j < curr->size; j++) {
      for (int k = 0; k < next->size; k++) {
        curr->weights[j][k] += nn->learning_rate * curr->deltas_weights[j][k];
      }
      curr->biases[i] += nn->learning_rate * curr->deltas_biases[j];
    }
  }
}

static void backprop(NeuralNetwork *nn, float reward) {
  for (size_t i = nn->size - 1; i > 0; i--) {
    NNLayer *curr = nn->layers[i];
    NNLayer *prev = nn->layers[i - 1];
    for (int j = 0; j < curr->size; j++) {
      // curr->deltas[]
      for (int k = 0; k < prev->size; k++) {
        prev->deltas_weights[k][j] = curr->deltas[j] * prev->outputs[i];
      }
      curr->deltas_biases[i] += curr->deltas[i];
    }
  }
}

void get_deltas(NeuralNetwork *nn, NNHistory *h, float reward) {
  forward_pass(nn, h->input);
  NNLayer *out = nn->layers[nn->size - 1];
  for (size_t i = 0; i < out->size; i++) {
    float s = (uint8_t)i == h->taken ? 1 : 0;
    out->deltas[i] += (s - out->outputs[i]) * reward;
  }
}

void nn_train(NeuralNetwork *nn, float reward) {
  // Esegui la back propagation
  NNHistory *curr = nn->history;
  while (curr) {
    get_deltas(nn, curr, reward);
    curr = curr->next;
  }
}
