import math
import numpy as np
import pandas

class Neuron:
  def __init__(self, inputs):
    self.weights = np.random.uniform(low=0.0, high=1.0, size=len(inputs))
    self.output = np.zeros(len(inputs))

  def activation(self, inputs):
    return sigmoid(np.sum(inputs * self.weights))


# Funções auxiliares
def sigmoid(x):
  return 1 / (1 + math.exp(-x))