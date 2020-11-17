import argparse
import neural_net as nn

parser = argparse.ArgumentParser(description='Backpropagation Neural Network.')
parser.add_argument('network', metavar='network', type=str,
                   help='Arquivo indicando a estrutura da rede. A primeira linha do \
                   arquivo deve armazenar o fator de regularização λ a ser utilizado. \
                   Cada linha subsequente no arquivo representa uma camada e o valor \
                   numérico armazenado naquela linha indica o número de neurônios na \
                   camada correspondente. Note que a primeira camada corresponde à \
                   camada de entrada da rede e a última camada corresponde à camada \
                   de saídas.')
parser.add_argument('initial_weights', metavar='weights', type=str,
                   help=' Arquivo indicando os pesos iniciais a serem utilizados pela \
                   rede. Cada linha do arquivo armazena os pesos iniciais dos neurônios \
                   de uma dada camada—isto é, os pesos dos neurônios da camada i são \
                   armazenados na linha i do arquivo. Os pesos de cada neurônio devem \
                   ser separados por vírgulas, e os neurônios devem ser separados por \
                   ponto-e-vírgula. Note que o primeiro peso de cada neurônio corresponde \
                   ao peso de bias.')
parser.add_argument('dataset', metavar='data', type=str,
                   help='Arquivo com um conjunto de treinamento. Cada linha representa \
                   uma instância e cada coluna representa um atributo. Após listados \
                   todos os valores dos atributos, há um ponto-e-vírgula, e os demais \
                   valores na linha indicam as saídas da rede para aquela instância.')

args = parser.parse_args()

# Faz o parse do arquivo descritor da rede.
with open(args.network) as network_file:
  aux = network_file.read().splitlines()

lamb = aux.pop(0)
network = [int(i) for i in aux]

# Faz o parse do arquivo que contém os pesos iniciais.
initial_weights = []
with open(args.initial_weights) as weights_file:
  line = weights_file.readline()
  while line != '':
    initial_weights.append([list(map(float, layer.split(','))) for layer in line.split(';')])
    line = weights_file.readline()

# Obtém o arquivo contendo o dataset. Este precisa ser fechado ao final do programa.
dataset = []
with open(args.dataset) as dataset_file:
  line = dataset_file.readline()
  while line != '':
    dataset.append([list(map(float, instance.split(','))) for instance in line.split(';')])
    line = dataset_file.readline()

# Roda a rede neural com os parametros obtidos.
net = nn.NeuralNetwork(network, initial_weights, 0.05, 0.0)

net.train(dataset)

print('Network: \n{}\n'.format(net.network))

print('Weights:')
print(net.weights)

print('Deltas:')
print(net.deltas)

print('Gradients:')
print(net.gradients)

print('Outputs:')
print(net.outputs)