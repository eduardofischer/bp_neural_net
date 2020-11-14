import math
import numpy as np
import pandas
from graphviz import Digraph

# Rede Neural
# Parâmetros:
# - network: lista contendo o número de neurônios em cada camada
# Ex.: [2, 3, 1] representa uma rede com 2 entradas, uma camada intermediária com 3 neurônios e 1 saída
# - initial_weights: lista de camadas, sendo cada camada uma lista de pesos dos neurônios
# Ex.: [[0.2, 0.3], [0.5, 0.4, 0.3], [0.8]]
# - alpha: taxa de aprendizagem
class NeuralNetwork:
  def __init__(self, network, initial_weights=None, alpha=0.05):
    # A rede é armazenada como uma lista de camadas, sendo cada camada uma lista de neurônios
    # Um determinado neurônio pode ser endereçado por self.network[layer][neuron]
    self.network = network
    self.alpha = alpha
    self.weights = []
    self.deltas = []
    self.outputs = []

    # Inicializa a rede neural
    for layer in range(len(network)):
      self.weights.append([])
      n_inputs = 1 if layer == 0 else network[layer - 1]
      for neuron in range(network[layer]):
        input_weights = initial_weights[layer][neuron] if initial_weights else np.random.uniform(low=0.0, high=1.0, size=n_inputs)
        self.weights[layer].append(input_weights)

  # TODO: Re-calcula os pesos da rede
  def _update_weights(self):
    pass

  # TODO: Re-calcula os deltas da rede
  def _update_deltas(self, expected_output):
    for layer in reversed(range(len(network))):
      if layer < len(self.network) - 1:
        # Para todas as camadas, exceto a última
        pass
      else:
        # Para a última camada
        # deltas.append()
        pass
  
  # TODO: Re-calcula as saídas dos neurônios
  def _update_outputs(self):
    pass

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
            weight = self.weights[layer + 1][neuron_next][neuron]
            label = str("{:.3f}".format(weight))
            dot.edge(get_label(layer, neuron), get_label(layer + 1, neuron_next), label=None)
    display(dot)

## Métodos Auxiliares

# Função sigmóide, mapeia um valor x para o intervalo de 0 a 1
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Calcula a ativação de um neurônio
def _activation(inputs, weights):
  return sigmoid(np.sum(inputs * weights))

# Gera uma label para um determinado neurônio
def get_label(layer, neuron):
  return 'l{0}n{1}'.format(layer, neuron)

  