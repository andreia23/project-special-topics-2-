from treeD import Arvore
from k_nN import KnN
from mlp import Mlp
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
            run("wine", "inicial", 13)
        elif optionDataset in ("2"): 
            run("balance-scale", "inicial", 4)
        elif optionDataset in ("3"): 
            run("abalone", "inicial", 8)
        elif optionDataset in ("4"): 
            run("iris", "final", 4)
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
    arvore_1.treinamento_resultado('entropy')
    
    knN_1 = KnN(randomBase)
    knN_1.treinamento_resultado('euclidean', 5)
    
    knN_2 = KnN(randomBase)
    knN_2.treinamento_resultado('euclidean', 10)

    mlp_1 = Mlp(randomBase)
    mlp_1.treinamento_resultado('tanh')

    mlp_2 = Mlp(randomBase)
    mlp_2.treinamento_resultado('logistic')

    print(f"Arvore_1: {arvore_1}\n")
    
    print(f"KnN_1: {knN_1}\n")
    print(f"KnN_2: {knN_2}\n")
    
    print(f"MLP_1: {mlp_1}\n")
    print(f"MLP_2: {mlp_2}\n")

if __name__ == '__main__':
    main()