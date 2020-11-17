import argparse

parser = argparse.ArgumentParser(description='Backpropagation Neural Network.')
parser.add_argument('network', metavar='network', type=str,
                   help='Arquivo indicando a estrutura da rede. A primeira linha do arquivo deve armazenar o fator de regularização λ a ser utilizado. Cada linha subsequente no arquivo representa uma camada e o valor numérico armazenado naquela linha indica o número de neurônios na camada correspondente. Note que a primeira camada corresponde à camada de entrada da rede e a última camada corresponde à camada de saídas.')
parser.add_argument('initial_weights', metavar='weights', type=str,
                   help=' Arquivo indicando os pesos iniciais a serem utilizados pela rede. Cada linha do arquivo armazena os pesos iniciais dos neurônios de uma dada camada—isto é, os pesos dos neurônios da camada i são armazenados na linha i do arquivo. Os pesos de cada neurônio devem ser separados por vírgulas, e os neurônios devem ser separados por ponto-e-vírgula. Note que o primeiro peso de cada neurônio corresponde ao peso de bias.')
parser.add_argument('dataset', metavar='data', type=str,
                   help='Arquivo com um conjunto de treinamento. Cada linha representa uma instância e cada coluna representa um atributo. Após listados todos os valores dos atributos, há um ponto-e-vírgula, e os demais valores na linha indicam as saídas da rede para aquela instância.')

args = parser.parse_args()

# Parse network description file.
with open(args.network) as network_file:
  network = network_file.read().splitlines()

print(network)

# Parse initial weights file.
