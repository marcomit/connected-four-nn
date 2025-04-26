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

float sum(float *inputs, size_t size) {
  float res = 0;
  for (size_t i = 0; i < size; i++) {
    res += inputs[i];
  }
  return res;
}

uint8_t max(float *inputs, size_t size) {
  uint8_t res = 0;
  for (size_t i = 0; i < size; i++) {
    inputs[res] < inputs[i] && (res = i);
  }
  return res;
}

// Activation functions
float relu(float x) { return x > 0 ? x : 0; }
// float leaky_relu(float x) { return x > 0 ? x : 0.01f * x; }
// float elu(float x) { return x > 0 ? x : 1 * expf(-x); }
// float sigmoid(float x) { return 1.0f / (float)(1.0f + expf(-x)); }
// float swish(float x) { return x * sigmoid(x); }

// Derivative activation functions
// float derivative_relu(float x) { return x > 0 ? 1.0f : 0; }
// float derivative_leaky_relu(float x) { return x > 0 ? 1 : 0.01f; }
// float derivative_elu(float x) { return x > 0 ? 1 : -expf(-x); }
// float derivative_sigmoid(float x){return logf(1.0f + )}

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

// void *activation(Activation act) {
//   void *func[6] = {relu, leaky_relu, elu, sigmoid, swish, NULL};
//   return func[act];
// }

// void *derivate_activation(Activation act) {
//   void *func[6] = {};
//   return func[act];
// }

NNLayer *dense(size_t size, float (*activation)(float),
               void (*normalize)(NNLayer *)) {
  NNLayer *l = malloc(sizeof(NNLayer));
  l->activation = activation;
  l->normalize = normalize;
  l->size = size;
  l->weights = NULL;

  l->biases = (float *)calloc(l->size, sizeof(float));
  l->outputs = (float *)calloc(l->size, sizeof(float));
  l->deltas = (float *)calloc(l->size, sizeof(float));

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
    if (prev->activation) {
      output = prev->activation(s);
    }
    next->outputs[i] = output;

    if (next->normalize) {
      next->normalize(next);
    }
  }
}

void forward_pass(NeuralNetwork *nn, float *inputs) {
  memcpy(nn->layers[0]->biases, inputs, nn->layers[0]->size);
  for (int i = nn->size - 1; i > 1; i--) {
    forward_pass_layer(nn, i - 1);
  }
}

uint8_t nn_run(NeuralNetwork *nn, GameState *state) {
  forward_pass(nn, (float *)state->board);
  NNLayer *output = nn->layers[nn->size - 1];
  for (int i = 0; i < COLS; i++) {
    printf("%d, ", (int)(output->outputs[i] * 100));
  }
  printf("\n");
  uint8_t res = max(output->outputs, output->size);
  // Se res non e' valida allora faccio subito backprop con un reward negativo.
  return res;
}

static void nn_init_weights(NeuralNetwork *nn) {
  for (size_t i = 0; i < nn->size - 1; i++) {
    nn->layers[i]->weights =
        (float **)malloc(sizeof(float *) * nn->layers[i]->size);

    for (size_t j = 0; j < nn->layers[i]->size; j++) {
      nn->layers[i]->weights[j] =
          (float *)malloc(sizeof(float) * nn->layers[i + 1]->size);
    }
  }
}

NeuralNetwork *nn_create(size_t size, float learning_rate) {
  printf("creazione rete\n");
  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->layers = (NNLayer **)malloc(sizeof(NNLayer *) * size);
  nn->size = size;
  nn->learning_rate = learning_rate;

  return nn;
}

static void backprop(NeuralNetwork *nn, float reward) {
  // Passi da fare.
  //

  // Calcolare i gradienti in base alle probabilita'
  // In base al reward aggiornare i gradienti.
}

void print_layer(NNLayer layer) { printf("layer: %zu", layer.size); }
void nn_train(uint16_t games) {
  printf("inizio train\n");
  NeuralNetwork *nn = nn_create(5, 0.1);

  nn->layers[0] = dense(42, relu, NULL);
  nn->layers[1] = dense(128, relu, NULL);
  nn->layers[2] = dense(64, relu, NULL);
  nn->layers[3] = dense(32, relu, NULL);
  nn->layers[4] = dense(7, NULL, softmax);

  nn_init_weights(nn);

  GameState *state = game_init();
  float rewards[3] = {-0.1f, 1.0f, -1.0f};
  for (int i = 0; i < games; i++) {
    game_loop(state, nn);
    // float res = rewards[*game_loop(state, nn)];
    // backprop(nn, res);
  }
}
