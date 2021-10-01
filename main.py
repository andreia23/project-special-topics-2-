from treeD import Arvore
from k_nN import KnN
from mlp import Mlp
from k_means import Kmean
from datasets_ import Dataset
from trainTest import Train_test
import pandas as pd

#Parameters: Name, Y_position, Number of Attributes
#Dataset("wine", "inicial", 13) 
#Dataset("balance-scale", "inicial", 4) 

def main():

    while True:

        optionDataset = input(
            "\nChoose an option: \n" + 
            "1- Wine\n" + 
            "2- Balance-scale\n" + 
            "3- Abalone\n" + 
            "0- Exit\n" + "\n>> ")
        if optionDataset in ("1"):
            run("wine", "inicial", 13) # Classes:3 (1,2,3)
        elif optionDataset in ("2"): 
            run("balance-scale", "inicial", 4) # Classes:3 (L,B,R)
        elif optionDataset in ("3"): 
            run("abalone", "inicial", 8) # Classes:3 (M, F, I)
        elif optionDataset in ("4"): 
            run("iris", "final", 4) # Classes:3 (1-Iris Setosa, 2-Iris Versicolour, 3-Iris Virginica)
        elif optionDataset in ("5"): 
            run("tic-tac-toe ", "final", 9)
        elif optionDataset in ("6"): 
            run("synchronous machine", "inicial", 4)
        elif optionDataset in ("0"):
            break
        else:
            print("Invalid Option!") 
    
def run(name, y_position, numberAttributes):

    dataset = Dataset(name, y_position, numberAttributes)
    randomBase = Train_test(dataset)
    randomBase.eighty_by_twenty()

    arvore_1 = Arvore(randomBase)
    arvore_1.treinamento_resultado('entropy') # Critério
    
    knN_1 = KnN(randomBase)
    knN_1.treinamento_resultado('euclidean', 5) # Distância | Vizinhança
    
    knN_2 = KnN(randomBase)
    knN_2.treinamento_resultado('euclidean', 10)

    mlp_1 = Mlp(randomBase)
    mlp_1.treinamento_resultado((5,3)) # Número de Neurônios por Camadas

    mlp_2 = Mlp(randomBase)
    mlp_2.treinamento_resultado((3,8,16)) 
    
    kmeans = Kmean(randomBase)
    kmeans.treinamento_resultado()

    print(f"Arvore_1: {arvore_1}\n")
    
    print(f"KnN_1: {knN_1}\n")
    print(f"KnN_2: {knN_2}\n")
    
    print(f"MLP_1: {mlp_1}\n")
    print(f"MLP_2: {mlp_2}\n")

    print(f"Kmeans: {kmeans}\n")

if __name__ == '__main__':
    main()