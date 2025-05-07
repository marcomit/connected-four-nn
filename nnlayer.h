#ifndef NNLAYER_H
#include <stddef.h>
typedef struct NeuralNetworkLayer NeuralNetworkLayer;

typedef enum {
  LAYER_DENSE,
  LAYER_CNN,
  LAYER_RNN,
  LAYER_POOLING,
  LAYER_GRU,
  LAYER_LSTM
} NeuralNetworkLayerType;

typedef enum {
  // Ridge functions
  ACTIVATION_IDENTITY,
  ACTIVATION_RELU,
  ACTIVATION_SIGMOID,
  ACTIVATION_SWISH,
  ACTIOVATION_TANH,
  ACTIVATION_SOFTSIGN,

  ACTIOVATION_SOFTPLUS,
  ACTIVATION_ELISH,
  ACTIVATION_SINUSOID,
  ACTIVATION_GAUSSIAN,

  ACTIVATION_SOFTMAX,
  ACTIVATION_MAXOUT
  // Radial functions
} NeuralNetworkActivation;

typedef NeuralNetworkActivation NNA;

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
} NeuralNetworkDenseParams;

typedef struct {
  size_t in_ch;
  size_t in_w;
  size_t in_h;

  size_t out_ch;
  size_t out_w;
  size_t out_h;

  size_t kernel_len;
  size_t stride;
  size_t padding;

  float *kernels;
  float *b;
  float *d_kernels;
  float *db;
  float *col_buffer;
  float *input_cache;

  NNA activation;
} NeuralNetworkConv2DParams;

typedef struct {
  size_t in_len;
  size_t hidden_len;

  float *W_f;
  float *W_i;
  float *W_o;
  float *W_c;

  float *U_f;
  float *U_i;
  float *U_o;
  float *U_c;

  float *b_f;
  float *b_i;
  float *b_o;
  float *b_c;

  float *h_prev;
  float *c_prev;

  float *f_gate;
  float *i_gate;
  float *o_gate;
  float *c_tilde;
  float *c_states;
  float *h_states;
  float *inputs;

  size_t seq_len;
  size_t current_batch_size;
} NeuralNetworkLSTMParams;

typedef NeuralNetworkLayer NNL;

/**
 * @struct NeuraNetworkLayer
 * @brief Represent the generic structure of a neural network's layer
 * */
typedef struct NeuralNetworkLayer {
  NeuralNetworkLayerType type; ///< Type of the layer e.g. DENSE, CNN, RNN etc.
  void *params; ///< The data of the layer that depends on the its type

  void (*forward)(void *, float *, size_t);  /// Forward pass
  void (*backward)(void *, float *, size_t); /// Back propagation
} NeuralNetworkLayer;

NNL *dense(size_t, NNA);
NNL *conv2D(size_t, NNA);
NNL *lstm();
NNL *rnn(size_t, NNA);
NNL *gru(size_t, NNA);

#endif // !NNLAYER_H
