import math
import numpy as np
import pandas
from graphviz import Digraph

# Rede Neural
# Parâmetros:
# - network: lista contendo o número de neurônios em cada camada
# Ex.: [2, 3, 1] representa uma rede com 2 entradas, uma camada intermediária com 3 neurônios e 1 saída
# - initial_weights: lista de camadas, sendo cada camada uma lista de pesos dos neurônios (incluindo os pesos de bias)
# Ex.: [[0.2, 0.3, 0.5], [0.5, 0.4, 0.3, 0.6], [0.8]]
# - alpha: taxa de aprendizagem
class NeuralNetwork:
  def __init__(self, network, initial_weights=None, alpha=0.05):
    self.network = network
    self.alpha = alpha
    self.deltas = [[] for x in range(len(network))]
    self.outputs = [[] for x in range(len(network))]

    if initial_weights:
      self.weights = initial_weights
    else:
      # Inicializa a rede com pesos aleatórios
      self.weights = [[] for x in range(len(network))]
      for layer in range(len(network)):
        n_inputs = 1 if layer == 0 else network[layer - 1]
        # Inclui o peso de bias caso a camada não seja a primeira (input)
        if layer is not 0:
          n_inputs = n_inputs + 1
        # Gera um conjunto de pesos de entrada para cada neurônio da camada
        for neuron in range(network[layer]):
          self.weights[layer].append(np.random.uniform(low=0.0, high=1.0, size=n_inputs))

  # TODO: Calcula os pesos da rede
  def _update_weights(self):
    pass

  # Calcula os deltas da rede
  def _update_deltas(self, expected_output):
    for layer in reversed(range(len(self.network))):
      # Se não for a camada de output
      if layer < len(self.network) - 1:
        for neuron in range(self.network[layer]):
          output_weights = [item[neuron + 1] for item in self.weights[layer + 1]]
          output_deltas = self.deltas[layer + 1]
          activation = self.outputs[layer][neuron]
          delta = np.sum(np.multiply(output_weights, output_deltas))*activation*(1-activation)
          self.deltas[layer].append(delta)
      else: # Para a camada de output
        self.deltas[layer].append(self.outputs[layer] - expected_output)
  
  # Calcula as saídas (ativação) dos neurônios
  def _update_outputs(self, inputs):
    for layer in range(len(self.network)):
      if layer == 0:
        input_weights = [item[0] for item in self.weights[layer]]
        self.outputs[layer] = input_weights * np.array(inputs)
      else:
        neuron_inputs = np.insert(self.outputs[layer - 1], 0, 1)
        self.outputs[layer] = np.array([activation(neuron_inputs, neuron_weights) for neuron_weights in self.weights[layer]])

  # Treina a rede neural
  def train(self, inputs_array, outputs_array):
    for inputs, outputs in zip(inputs_array, outputs_array):
      self._update_outputs(inputs)
      self._update_deltas(outputs)

  # Plota um grafo representando a rede neural
  def plot(self):
    dot = Digraph()
    dot.attr(rankdir='LR', splines='false', ranksep='.8', nodesep='.4', labelloc='b')
    dot.node_attr['shape'] = 'circle'
    dot.edge_attr['arrowsize'] = '.5'
    dot.edge_attr['penwidth'] = '.5'
    dot.node_attr['margin'] = '0'
    for layer in range(len(self.network)):
      for neuron in range(self.network[layer]):
        if layer < len(self.network) - 1:
          # Se não for a última camada
          for neuron_next in range(self.network[layer + 1]):
            weights1 = self.weights[layer][neuron]
            weights2 = self.weights[layer + 1][neuron_next]
            dot.edge('\n'.join(format(x, "^.3f") for x in weights1), '\n'.join(format(x, "^.3f") for x in weights2))
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
  return 'l{0}n{1}'.format(layer, neuron)


  