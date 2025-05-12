#include "ann.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef NeuralNetworkLayer NNL;

typedef float (*LossFunc)(float *, float *, size_t);

float nnmax(float *X, size_t len) {
  float m = X[0];
  for (size_t i = 1; i < len; i++) {
    if (X[i] > m)
      m = X[i];
  }
  return m;
}

float sum(float *X, size_t len) {
  float s = 0.0f;
  for (size_t i = 0; i < len; i++) {
    s += X[i];
  }
  return s;
}

float *mul(float *a, float *b, size_t len) {
  float *out = malloc(sizeof(float) * len);
  for (size_t i = 0; i < len; i++) {
    out[i] = a[i] * b[i];
  }
  return out;
}

// Activation functions
// Ridge functions
float absf(float x) { return x < 0 ? -x : x; }
float ReLU(float x) { return x > 0 ? x : 0; }
float identity(float x) { return x; }
float sigmoid(float x) { return 1 / (1 + expf(x)); }
float nntanh(float x) { return tanhf(x); }
float softsign(float x) { return x / (1 + absf(x)); }
float softplus(float x) { return logf(1 + expf(x)); }
float leaky_relu(float x) {
  float alpha = 0.01f;
#ifdef LEAKY_RELU_ALPHA
  alpha = LEAKY_RELU_ALPHA;
#endif
  return ReLU(x) * alpha;
}

float swish(float x) { return x / (1 + expf(-x)); }
float ELiSH(float x) {
  if (x < 0) {
    return (expf(x) - 1) / (1 + expf(-x));
  }
  return x / (1 + expf(-x));
}
float gaussian(float x) { return expf(-powf(x, 2.0f)); }
float sinusoid(float x) { return sinf(x); }

// Derivate activation functions
float dReLU(float x) { return x > 0 ? 1 : 0; }
float didentity(float x) { return 1; }
float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
float dtanh(float x) { return 1 - powf(nntanh(x), 2); }
float dsoftsign(float x) { return 1 / powf((1 + absf(x)), 2); }
float dsoftplus(float x) { return 1 / (1 + expf(-x)); }
float dswish(float x) {
  float n = 1 + expf(-x) + x * expf(-x);
  float d = powf(1 + expf(-x), 2);
  return n / d;
}
float dELiSH(float x) {
  float n;
  if (x < 0) {
    n = 2 * expf(2 * x) + expf(3 * x) - expf(x);
  } else {
    n = x * expf(x) + expf(2 * x) + expf(x);
  }
  return n / expf(2 * x) + 2 * expf(x) + 1;
}
float dgaussian(float x) { return -2 * expf(-powf(x, 2)); }
float dsinusoid(float x) { return cosf(x); }

// End activation functions

static float activation_single(NNA a, float x) {
  float (*func[10])(float) = {identity, ReLU,     sigmoid, swish,    nntanh,
                              softsign, softplus, ELiSH,   sinusoid, gaussian};
  return func[a](x);
}

static float activation_derivative_single(NNA type, float x) {
  float (*func[10])(float) = {didentity, dReLU,     dsigmoid,  dswish,
                              dtanh,     dsoftsign, dsoftplus, dELiSH,
                              dsinusoid, dgaussian};
  return func[type](x);
}

static void activation(NNL *layer) {
  NNA activation = layer->activation;
  if (activation == SOFTMAX) {
    float m = nnmax(layer->z, layer->len);
    float s = 0.0f;
    for (size_t i = 0; i < layer->len; i++) {
      layer->a[i] = expf(layer->z[i] - m);
      s += layer->a[i];
    }
    for (size_t i = 0; i < layer->len; i++) {
      layer->a[i] /= s;
    }
    return;
  }
  if (activation == MAXOUT) {
    return;
  }
  for (int i = 0; i < layer->len; i++) {
    layer->a[i] = activation_single(layer->activation, layer->z[i]);
  }
}

static void activation_derivative(NNL *layer, float *out) {
  NNA activation = layer->activation;
  if (activation == SOFTMAX) {
    return;
  }
  if (activation == MAXOUT) {
    return;
  }
  for (int i = 0; i < layer->len; i++) {
    out[i] = activation_derivative_single(layer->activation, layer->z[i]);
  }
}

// Cost functions

// Mean squared error
// It is used for regression problems.
// It calculates the average squared difference between predicated and actual
// values
static float mean_squared_error(float *outputs, float *targets, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    sum += powf(targets[i] - outputs[i], 2);
  }
  return sum / len;
}

// Binary cross entropy loss
// It is used for binary classification problems.
// It measures the difference between the predicated and actual probabilities
// of the binary output available
static float binary_ce(float *outputs, float *targets, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    sum +=
        outputs[i] * logf(targets[i]) + (1 - outputs[i]) * logf(1 - targets[i]);
  }
  return -sum / len;
}

// Category cross entropy
// It is used for Multi-class classification problems
// It measures the difference between the predicates and actual probabilities
// of multi-classes output available
static float category_ce(float *outputs, float *targets, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    sum += outputs[i] * logf(targets[i]);
  }
  return -sum / len;
}

// Hinge loss
// Commonly used for binary classification tasks
// using Support Vector Machines (SVMs)
// It penalizes misclassified samples and aim to maximize
// the margin between the decision boundary and the training samples
static float hinge_loss(float *outputs, float *targets, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    sum += outputs[i] * targets[i];
  }
  sum = 1 - sum;
  if (sum < 0)
    return 0;
  return sum;
}

// Kullback-Leibler Divergence
// It measures the differencebetween predicated and actual probability
// distributions. It is commonly used in tasks such as generative modeling
static float kl_divergence(float *outputs, float *targets, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    sum += outputs[i] * logf(outputs[i] / targets[i]);
  }
  return sum;
}

// Root Mean Squared error
// This is a variant of MSE.
// It is used to measure the performance of regression models
// when the scale of the target variable matters
static float root_mse(float *outputs, float *targets, size_t len) {
  return sqrtf(mean_squared_error(outputs, targets, len));
}

NeuralNetwork *nncreate(size_t len, float learning_rate) {
  NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  nn->len = len;
  nn->layers = (NNL **)malloc(sizeof(NNL *) * len);
  nn->learning_rate = learning_rate;
  return nn;
}

// static float *initf(size_t len) { return (float *)calloc(len,
// sizeof(float));
// }

NNL *nndense(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));

  layer->len = len;

  layer->a = calloc(len, sizeof(float) * len);
  layer->b = calloc(len, sizeof(float) * len);
  layer->z = calloc(len, sizeof(float) * len);

  layer->d = calloc(len, sizeof(float) * len);
  layer->db = calloc(len, sizeof(float) * len);

  layer->activation = activation;
  return layer;
}

NNL *cnn(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));
  return layer;
}

NNL *nnpooling(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));
  return layer;
}

NNL *rnn(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));
  return layer;
}

NNL *nngru(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));
  return layer;
}

NNL *nnlstm(size_t len, NNA activation) {
  NNL *layer = (NNL *)malloc(sizeof(NNL));
  return layer;
}

void nninit(NeuralNetwork *nn) {
  for (size_t i = 1; i < nn->len; i++) {
    NNL *curr = nn->layers[i];
    NNL *prev = nn->layers[i - 1];
    curr->W = (float **)malloc(sizeof(float *) * curr->len);
    curr->dW = (float **)malloc(sizeof(float *) * curr->len);
    float scale = sqrtf(6.0f / (prev->len + curr->len));

    for (size_t j = 0; j < curr->len; j++) {
      curr->W[j] = (float *)calloc(prev->len, sizeof(float));
      curr->dW[j] = (float *)calloc(prev->len, sizeof(float));
      curr->b[j] = 0.01f * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
      for (size_t k = 0; k < prev->len; k++) {
        curr->W[j][k] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
      }
    }
  }
}

static void forward_pass_layer(NNL *prev, NNL *curr) {
  for (size_t i = 0; i < curr->len; i++) {
    float z = curr->b[i];
    for (size_t j = 0; j < prev->len; j++) {
      z += prev->a[j] * curr->W[i][j];
    }
    curr->z[i] += z;
    curr->a[i] += z;
  }
  activation(curr);
}

void nnforesee(NeuralNetwork *nn, float *inputs) {
  // Copy the input into the first layer
  for (size_t i = 0; i < nn->layers[0]->len; i++) {
    nn->layers[0]->z[i] = inputs[i];
  }

  for (size_t i = 1; i < nn->len; i++) {
    NNL *prev = nn->layers[i - 1];
    NNL *curr = nn->layers[i];
    forward_pass_layer(prev, curr);
  }
}

static void backward_output(NNL *out, float *target) {
  for (size_t i = 0; i < out->len; i++) {
    float dz = out->a[i] - target[i];

    if (out->activation == SOFTMAX) {
      out->d[i] = dz;
    } else {
      float da = activation_derivative_single(out->activation, dz);
      out->d[i] = da * dz;
    }
  }
}

static void backward_hidden(NNL *curr, NNL *next) {
  // printf("backward\n\n");
  for (size_t i = 0; i < curr->len; i++) {
    float s = 0.0f;
    for (size_t j = 0; j < next->len; j++) {
      s += next->W[j][i] * next->d[j];
    }
    float da = activation_derivative_single(curr->activation, curr->z[i]);
    curr->d[i] = s * da;
  }
  // printf("fine\n");
}

static void compute_gradients(NNL *curr, NNL *prev) {
  for (size_t i = 0; i < curr->len; i++) {
    for (size_t j = 0; j < prev->len; j++) {
      curr->dW[i][j] = curr->d[i] * prev->z[j];
    }
  }
}

static void apply_gradients(NNL *curr, NNL *prev, float learning_rate) {
  for (size_t i = 0; i < curr->len; i++) {
    for (size_t j = 0; j < prev->len; j++) {
      curr->W[i][j] -= learning_rate * curr->dW[i][j];
    }
    curr->b[i] -= learning_rate * curr->db[i];
  }
}

// For the last layer we calculate the deltas with the targets
void nnbalance(NeuralNetwork *nn, float *target) {
  size_t L = nn->len;
  backward_output(nn->layers[L - 1], target);

  for (size_t i = L - 2; i > 0; i--) {
    backward_hidden(nn->layers[i - 1], nn->layers[i]);
  }

  compute_gradients(nn->layers[1], nn->layers[0]);

  for (size_t i = 2; i < L; i++) {
    NNL *curr = nn->layers[i];
    NNL *prev = nn->layers[i - 1];
    compute_gradients(curr, prev);
  }

  for (size_t i = 1; i < L; i++) {
    NNL *curr = nn->layers[i];
    NNL *prev = nn->layers[i - 1];
    apply_gradients(curr, prev, nn->learning_rate);
  }
}

float *nnforward(NeuralNetwork *net, float *inputs) {
  nnforesee(net, inputs);
  NNL *out = net->layers[net->len - 1];
  return out->a;
}

/* La prima riga e' l'intestazione della rete.
 * Cioe' il primo numero e' il numero di layer ed il secondo il learning_rate.
 *
 * Caricamento dei layer:
 * Ogni layer ha la seguente struttura:
 *  - Prima riga  : contiene il numero di neuroni e la funzione di attivazione
 *  - Seconda riga: contiene la lista di biases del layer
 *  - Terza riga  : contiene i pesi ed avra una lunghezza pari a:
 *    lunghezza del layer x la lunghezza del layer precedente (Il primo layer
 * non ha pesi).
 */
void nnsave(NeuralNetwork *nn, const char *f) {
  FILE *file = fopen(f, "w");

  fprintf(file, "%zu %.5f\n", nn->len, nn->learning_rate);

  // Carica tutte le dimensioni dei layer
  for (size_t i = 0; i < nn->len; i++) {
    fprintf(file, "%zu %d\n", nn->layers[i]->len, nn->layers[i]->activation);
  }

  // Carica i bias del primo layer
  for (size_t i = 0; i < nn->layers[0]->len; i++) {
    fprintf(file, "%.5f ", nn->layers[0]->b[i]);
  }
  fprintf(file, "\n");

  for (size_t i = 1; i < nn->len; i++) {
    NNL *curr = nn->layers[i];
    NNL *prev = nn->layers[i - 1];

    // Biases del layer
    for (size_t j = 0; j < curr->len; j++) {
      fprintf(file, "%.5f ", curr->b[j]);
    }
    fprintf(file, "\n");

    // Pesi del layer
    for (size_t j = 0; j < curr->len; j++) {
      for (size_t k = 0; k < prev->len; k++) {
        fprintf(file, "%.5f ", curr->W[j][k]);
      }
      fprintf(file, "\n");
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

static void load_layer(NNL *l, FILE *fd, size_t prev_len) {
  float v;
  fscanf(fd, "%f", &v);
  for (size_t i = 0; i < l->len; i++) {
    l->W[i] = malloc(sizeof(float) * prev_len);
    l->dW[i] = malloc(sizeof(float) * prev_len);
    // l->a[i] = 0;
    // l->d[i] = 0;
    // l->b[i] = 0;
    // l->z[i] = 0;
    for (size_t j = 0; j < prev_len; j++) {
      fscanf(fd, "%f", &v);
      l->W[i][j] = v;
      l->dW[i][j] = 0;
    }
  }
}

void nnload(NeuralNetwork *net, const char *f) {
  FILE *fd;

  if ((fd = fopen(f, "r")) == 0) {
    printf("File does not exists\n");
    return;
  }
  printf("Load network from file %s\n", f);

  float v;

  fscanf(fd, "%zu", &net->len);

  net->layers = malloc(sizeof(NNL *) * net->len);

  fscanf(fd, "%f", &v);

  net->learning_rate = v;

  for (size_t i = 0; i < net->len; i++) {
    net->layers[i] = malloc(sizeof(NNL));
    size_t len;
    NNA nna;

    fscanf(fd, "%zu", &len);
    fscanf(fd, "%d", &nna);

    net->layers[i]->activation = (NNA)nna;
    net->layers[i]->len = len;
    net->layers[i]->b = calloc(len, sizeof(float));
    net->layers[i]->a = calloc(len, sizeof(float));
    net->layers[i]->z = calloc(len, sizeof(float));
    net->layers[i]->d = calloc(len, sizeof(float));
    net->layers[i]->db = calloc(len, sizeof(float));

    net->layers[i]->W = malloc(sizeof(float *) * len);
    net->layers[i]->dW = malloc(sizeof(float *) * len);
  }

  for (size_t i = 0; i < net->layers[0]->len; i++) {
    fscanf(fd, "%f", &v);
    net->layers[0]->b[i] = v;
  }
  for (size_t i = 1; i < net->len; i++) {
    NNL *curr = net->layers[i];
    NNL *prev = net->layers[i - 1];
    load_layer(curr, fd, prev->len);
  }
}
