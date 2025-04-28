#include "neural_network.h"
#include "game.h"
#include <_stdio.h>
#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#define LEAKY_RELU_ALPHA 0.01
#define ELU_ALPHA 0.01

// Funzione per generare numeri casuali compresi tra -x e +x
#define RANDOM(x)                                                              \
  (((float)rand() / RAND_MAX) -                                                \
   0.5f) //(((2.0f * (x)) * ((float)rand() / (float)RAND_MAX)) - (x))

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
static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float leaky_relu(float x) {
  return x > 0 ? x : LEAKY_RELU_ALPHA * x;
}
static inline float elu(float x) { return x > 0 ? x : ELU_ALPHA * expf(-x); }
static inline float sigmoid(float x) { return 1.0f / (float)(1.0f + expf(-x)); }
static inline float swish(float x) { return x * sigmoid(x); }

// Derivative activation functions
static inline float drelu(float x) { return x > 0 ? 1.0f : 0.0f; }
static inline float dleaky_relu(float x) {
  return x > 0 ? 1 : LEAKY_RELU_ALPHA;
}
static inline float delu(float x) { return x > 0 ? 1 : -expf(-x); }
static inline float dsigmoid(float x) {
  return sigmoid(x) * (1 - sigmoid(x));
} // TODO
static inline float dswish(float x) { return x; } // TODO
void softmax(NNLayer *layer) {
  float *inputs = layer->outputs;
  int size = layer->size;

  float max_input = max(inputs, layer->size);

  float sum_exp = 0.0f;
  for (int i = 0; i < size; i++) {
    inputs[i] = expf(inputs[i] - max_input); // stabilità numerica
    sum_exp += inputs[i];
  }

  if (sum_exp == 0) {
    for (int i = 0; i < size; i++) {
      inputs[i] = 1.0f / size;
    }
    return;
  }
  for (int i = 0; i < size; i++) {
    inputs[i] /= sum_exp;
  }
}

float activation(Activation act, float in) {
  float (*func[6])(float) = {relu, leaky_relu, elu, sigmoid, swish, NULL};
  if (act == 5) {
    return in;
  }
  return func[act](in);
}

float derivate_activation(Activation act, float in) {
  float (*func[6])(float) = {drelu, dleaky_relu, delu, dsigmoid, dswish, NULL};
  if (act == 5) {
    return in;
  }
  return func[act](in);
}

NNLayer *dense(size_t size, Activation acttype, void (*normalize)(NNLayer *)) {
  NNLayer *l = malloc(sizeof(NNLayer));
  l->acttype = acttype;
  l->normalize = normalize;
  l->size = size;

  l->biases = (float *)calloc(l->size, sizeof(float));
  l->outputs = (float *)calloc(l->size, sizeof(float));
  l->deltas = (float *)calloc(l->size, sizeof(float));
  l->deltas_biases = (float *)calloc(l->size, sizeof(float));

  return l;
}

static void forward_pass_layer(NNLayer *curr, NNLayer *next) {
  float **w = curr->weights;
  for (size_t i = 0; i < next->size; i++) {
    float s = 0;
    for (size_t j = 0; j < curr->size; j++) {
      s += curr->outputs[j] * w[j][i];
    }
    s += next->biases[i];

    next->outputs[i] = activation(curr->acttype, s);
  }
  if (next->normalize) {
    next->normalize(next);
  }
}

static void forward_pass(NeuralNetwork *nn, float *inputs) {
  memcpy(nn->layers[0]->outputs, inputs, nn->layers[0]->size * sizeof(float));

  for (size_t i = 0; i < nn->size - 1; i++) {
    forward_pass_layer(nn->layers[i], nn->layers[i + 1]);
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
      curr->deltas_weights[j] = (float *)malloc(sizeof(float) * next->size);
      for (size_t k = 0; k < next->size; k++) {
        curr->weights[j][k] = RANDOM(0.5f);
        curr->deltas_weights[j][k] = 0.0f;
      }
    }
  }
}

void nn_free_history(NeuralNetwork *nn) { nn->num_moves = 0; }

uint8_t choose_stochastic(float *probs, size_t size) {
  float r = (float)rand() / RAND_MAX;
  float cum = 0.0f;
  for (uint8_t i = 0; i < size; i++) {
    cum += probs[i];
    if (r < cum) {
      return i;
    }
  }
  return size - 1; // fallback
}
void nn_save_move(NeuralNetwork *nn) {
  NNLayer *in = nn->layers[0];
  NNLayer *out = nn->layers[nn->size - 1];

  NNHistory history;
  history.input = malloc(sizeof(float) * in->size);
  memcpy(history.input, in->outputs, in->size * sizeof(float));
  history.taken = choose_stochastic(out->outputs, out->size);
  nn->history[nn->num_moves] = history;
  nn->num_moves++;
}

void nn_run(NeuralNetwork *nn, float *inputs) {
  forward_pass(nn, inputs);
  NNLayer *output = nn->layers[nn->size - 1];
  for (int i = 0; i < COLS; i++) {
    printf("%d, ", (int)(output->outputs[i] * 100));
  }
  printf("\n");
  nn_save_move(nn);
}

NeuralNetwork *nn_create(size_t size, float learning_rate, size_t max_moves) {

  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->layers = (NNLayer **)malloc(sizeof(NNLayer *) * size);
  nn->size = size;
  nn->learning_rate = learning_rate;

  nn->history = malloc(sizeof(NNHistory) * max_moves);
  nn->num_moves = 0;
  return nn;
}

static void update_weights(NeuralNetwork *nn) {
  // printf("update_weights\n");
  for (size_t i = 0; i < nn->size - 1; i++) {
    // printf("layer %zu\n", i);
    NNLayer *curr = nn->layers[i];
    NNLayer *next = nn->layers[i + 1];
    for (size_t j = 0; j < curr->size; j++) {
      // printf("neurone %zu ", j);
      for (size_t k = 0; k < next->size; k++) {
        curr->weights[j][k] += nn->learning_rate * curr->deltas_weights[j][k];
        curr->deltas_weights[j][k] = 0;
      }
      // printf("azzerati pesi\n");
      curr->biases[j] += nn->learning_rate * curr->deltas_biases[j];
      curr->deltas_biases[j] = 0;
    }
  }
}

static void backprop(NeuralNetwork *nn) {
  for (int64_t i = nn->size - 2; i >= 0; i--) {
    NNLayer *curr = nn->layers[i];
    NNLayer *next = nn->layers[i + 1];
    for (size_t j = 0; j < curr->size; j++) {

      float sum = 0;
      for (size_t k = 0; k < next->size; k++) {
        sum += next->deltas[k] * curr->weights[j][k];
      }

      curr->deltas[j] =
          sum * derivate_activation(curr->acttype, curr->outputs[j]);

      for (size_t k = 0; k < next->size; k++) {
        curr->deltas_weights[j][k] += curr->outputs[j] * next->deltas[k];
      }

      curr->deltas_biases[j] += curr->deltas[j];
    }
  }
}

float entropy(float *inputs, size_t size) {
  float res = 0.0f;
  for (size_t i = 0; i < size; i++) {
    res -= inputs[i] * logf(inputs[i]);
  }
  return res;
}

static void set_deltas(NeuralNetwork *nn, NNHistory h, float reward,
                       float move_importance) {
  forward_pass(nn, h.input);
  NNLayer *out = nn->layers[nn->size - 1];

  float loss = reward * move_importance;
  for (size_t i = 0; i < out->size; i++) {
    float s = (uint8_t)i == h.taken ? 1 : 0;
    out->deltas[i] += (s - out->outputs[i]) * loss;
  }
}

void nn_train(NeuralNetwork *nn, float reward) {
  printf("INIZIO Train--------------------------------\n");

  printf("PREMIO %f\n", reward);
  uint8_t moves = 0;
  for (int i = 0; i < nn->num_moves; i++) {
    float move_importance = 0.5f + 0.5f * (float)i / nn->num_moves;
    set_deltas(nn, nn->history[i], reward, move_importance);
    backprop(nn);
    moves++;
  }
  printf("layers\n");
  for (size_t i = 0; i < nn->layers[nn->size - 1]->size; i++) {
    printf("%d. %d\n", (int)i,
           (int)(nn->layers[nn->size - 1]->outputs[i] * 100));
  }
  printf("Hai giocato %d\n", moves);
  fflush(stdout);
  update_weights(nn);
  printf("FINE train -----------------------------\n");
}

static void nn_save(NeuralNetwork *nn, const char *filename) {
  FILE *file;
  file = fopen(filename, "w");
  perror("TODO");
}
