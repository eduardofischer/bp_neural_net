import math
import numpy as np
from graphviz import Digraph

# Rede Neural
# Parâmetros:
# - network: lista contendo o número de neurônios em cada camada
# Ex.: [2, 3, 1] representa uma rede com 2 entradas, uma camada intermediária com 3 neurônios e 1 saída
# - initial_weights: lista de camadas, sendo cada camada uma lista de pesos dos neurônios (incluindo os pesos de bias)
# Ex.: [[0.2, 0.3, 0.5], [0.5, 0.4, 0.3, 0.6], [0.8]]
# - alpha: taxa de aprendizagem
class NeuralNetwork:
  def __init__(self, network, initial_weights=None, alpha=0.05, lamb=0.25):
    self.network = network
    self.alpha = alpha
    self.lamb = lamb
    self.deltas = [[] for _ in np.empty([len(network) - 1])]
    self.gradients = [[] for _ in np.empty([len(network) - 1])]
    self.outputs = [[] for _ in np.empty([len(network)])]

    if initial_weights:
      self.weights = initial_weights
    else:
      # Inicializa a rede com pesos aleatórios
      self.weights = []
      for layer in range(len(network) - 1):
        self.weights.append([])
        # Número de entradas da camada + bias
        n_inputs = network[layer] + 1
        # Gera um conjunto de pesos de entrada para cada neurônio da camada
        for _ in range(network[layer + 1]):
          self.weights[layer].append(np.random.uniform(low=-1.0, high=1.0, size=n_inputs))

  # Calcula as saídas (ativação) dos neurônios
  def _update_outputs(self, inputs):
    for layer in range(len(self.network)):
      if layer == 0:
        self.outputs[layer] = np.array(inputs)
      else:
        # Insere o bias (1) nas entradas da camada
        neuron_inputs = np.insert(self.outputs[layer - 1], 0, 1)
        # Calcula o output dos neurônios da camada
        self.outputs[layer] = np.array([activation(neuron_inputs, neuron_weights) for neuron_weights in self.weights[layer - 1]])
  
  # Calcula os deltas da rede
  def _update_deltas(self, expected_output):
    for layer in reversed(range(len(self.network))):
      # Para as camadas intermediárias:
      if layer < len(self.network) - 1 and layer != 0:
        self.deltas[layer - 1] = []
        for neuron in range(self.network[layer]):
          output_weights = [item[neuron + 1] for item in self.weights[layer]]
          output_deltas = self.deltas[layer]
          activation = self.outputs[layer][neuron]
          delta = np.sum(np.multiply(output_weights, output_deltas))*activation*(1-activation)
          self.deltas[layer - 1] = np.append(self.deltas[layer - 1], delta)
      elif layer == len(self.network) - 1: # Para a camada de output
        self.deltas[layer - 1] = self.outputs[layer] - expected_output

  # Calcula os gradientes para os pesos da rede
  def _update_gradients(self):
    for layer in range(len(self.network)):
      if layer != 0:
        if len(self.gradients[layer - 1]) == 0:
          self.gradients[layer - 1] = [np.zeros(self.network[layer - 1] + 1) for _ in range(self.network[layer])]
        for neuron in range(self.network[layer]):
          origin_activations = np.insert(self.outputs[layer - 1], 0, 1)
          delta = self.deltas[layer - 1][neuron]
          self.gradients[layer - 1][neuron] = self.gradients[layer - 1][neuron] + (origin_activations * delta)

  def _regularize_gradients(self, n):
    P = [[] for _ in np.empty([len(self.network) - 1])]
    for layer in range(len(self.weights)):
      # Multiplica todos os pesos por lambda
      # Zera os pesos de bias, que não devem multiplicados por lambda
      P[layer] = np.asarray(self.weights[layer]) * self.lamb
      for neuron in range(len(self.weights[layer])):
        P[layer][neuron][0] = 0
      self.gradients[layer] = (self.gradients[layer] + P[layer])/n

  # Calcula os pesos da rede
  def _update_weights(self):
    for layer in range(len(self.weights)):
      self.weights[layer] = self.weights[layer] - self.alpha * np.asarray(self.gradients[layer])
      
  # Treina a rede neural
  def train(self, inputs_array, outputs_array, n_ephocs=1):
    for _ in range(n_ephocs):
      for inputs, exp_outputs in zip(inputs_array, outputs_array):
        self._update_outputs(inputs)
        self._update_deltas(exp_outputs)
        self._update_gradients()
      self._regularize_gradients(len(inputs_array))
      self._update_weights()

  def train2(self, dataset, n_ephocs=1):
    for _ in range(n_ephocs):
      for instance in dataset:
        self._update_outputs(instance[0])
        self._update_deltas(instance[1])
        self._update_gradients()
      self._regularize_gradients(len(dataset))
      self._update_weights()

  # Função de custo
  def cost(self, inputs_array, outputs_array):
    cost = 0
    for inputs, exp_outputs in zip(inputs_array, outputs_array):
      self._update_outputs(inputs)
      outputs = self.outputs[-1]
      J = -exp_outputs * math.log(outputs) - (1 - exp_outputs) * math.log(1 - outputs)
      cost = cost + J
    cost = cost / len(inputs_array)
    S = [[] for layer in self.weights]
    S_sum = (self.lamb/(2*len(inputs_array)))

    if self.lamb != 0.0:
      # Eleva todos os pesos ao quadrado
      # Restaura os pesos de bias, que não devem ser elevados ao quadrado
      for layer in range(len(self.weights)):
        S[layer] = np.asarray(self.weights[layer]) ** 2
        for neuron in range(len(self.weights[layer])):
          S[layer][neuron][0] = self.weights[layer][neuron][0]
          S_sum = S_sum + np.sum(S[layer][neuron])

    return cost + S_sum

  # Avalia a saída da rede para uma determinada entrada
  def classify(self, inputs, n_outputs=1):
    self._update_outputs(inputs)
    if n_outputs == 1:
      return 1 if self.outputs[-1][0] > .5 else 0

  # Plota um grafo representando a rede neural
  def plot(self):
    dot = Digraph()
    dot.attr(rankdir='LR', splines='false', ranksep='.8', nodesep='.4')
    dot.node_attr['shape'] = 'circle'
    dot.edge_attr['arrowsize'] = '.5'
    dot.edge_attr['penwidth'] = '.5'
    dot.node_attr['margin'] = '0'
    for layer in range(len(self.network)):
      for neuron in range(self.network[layer]):
        label1 = get_label(layer, neuron)
        if layer != 0:
          dot.edge('Bias {0}'.format(layer), label1, label='{:.3f}'.format(self.weights[layer - 1][neuron][0]))
        if layer != len(self.network) - 1:
          for neuron_next in range(self.network[layer + 1]):
            label2 = get_label(layer + 1, neuron_next)
            dot.edge(label1, label2, label='{:.3f}'.format(self.weights[layer][neuron_next][neuron + 1]))
    display(dot)

## Métodos Auxiliares

# Função sigmóide, mapeia um valor x para o intervalo de 0 a 1
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Calcula a ativação de um neurônio
def activation(inputs, weights):
  return sigmoid(np.sum(inputs * weights))

# Gera uma label para um determinado neurônio
def get_label(layer, neuron):
  return 'L {0}\nN {1}'.format(layer, neuron)


  