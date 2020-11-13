import math
import numpy as np
import pandas
from graphviz import Digraph

# Neurônio
# Parâmetros:
# - n_inputs: número de entradas do neurônio
# - weights: lista de camadas, sendo cada camada uma lista de pesos dos neurônios
# Ex.: [[0.2, 0.3], [0.5, 0.4, 0.3], [0.8]]
class Neuron:
  def __init__(self, n_inputs, weights=None):
    # Pesos para cada entrada do neurônio
    # - Caso não sejam fornecidos no construtor, os pesos são inicializados
    # com valores aleatórios entre 0 e 1
    self.weights = weights if weights else np.random.uniform(low=0.0, high=1.0, size=n_inputs)
    # Saída do neurônio (valor de ativação)
    self.output = None

  # Função de ativação do neurônio
  def _activation(self, inputs):
    return sigmoid(np.sum(inputs * self.weights))

  # Calcula a saída do neurônio para uma determinada entrada
  def update(self, inputs):
    self.output = self._activation(inputs)

# Rede Neural
# Parâmetros:
# - network: lista contendo o número de neurônios em cada camada
# Ex.: [2, 3, 1] representa uma rede com 2 entradas, uma camada intermediária com 3 neurônios e 1 saída
# - initial_weights: lista de camadas, sendo cada camada uma lista de pesos dos neurônios
# Ex.: [[0.2, 0.3], [0.5, 0.4, 0.3], [0.8]]
class NeuralNet:
  def __init__(self, network, initial_weights=None):
    # A rede é armazenada como uma lista de camadas, sendo cada camada uma lista de neurônios
    # Um determinado neurônio pode ser endereçado por self.network[layer][neuron]
    self.network = []

    # Inicializa a rede neural
    for layer in range(len(network)):
      self.network.append([])
      n_inputs = 1 if layer == 0 else network[layer - 1]
      for neuron in range(network[layer]):
        weights = initial_weights[layer][neuron] if initial_weights else None
        self.network[layer].append(Neuron(n_inputs, weights))

  def plot(self):
    dot = Digraph()
    dot.attr(rankdir='LR', splines='false')
    dot.edge_attr['fontsize'] = '8'
    dot.edge_attr['arrowsize'] = '.5'
    dot.edge_attr['penwidth'] = '.5'
    for layer in range(len(self.network)):
      for neuron in range(len(self.network[layer])):
        if layer < len(self.network) - 1:
          for neuron_next in range(len(self.network[layer + 1])):
            weight = self.network[layer + 1][neuron_next].weights[neuron]
            label = str("{:.3f}".format(weight))
            dot.edge(get_label(layer, neuron), get_label(layer + 1, neuron_next), label=None)
    display(dot)

# Métodos Auxiliares
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Gera uma label para um determinado neurônio
def get_label(layer, neuron):
  return 'l{0}n{1}'.format(layer, neuron)

  