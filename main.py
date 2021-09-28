from treeD import Arvore
from k_nN import KnN
from datasets_ import Dataset
from trainTest import Train_test
import pandas as pd

#Parameters: Name, Y_position, Number of Attributes
#Dataset("wine", "inicial", 13) 
#Dataset("iris", "final", 4) 
#Dataset("balance-scale", "inicial", 4) 

def main():

    while True:

        optionDataset = input("\nEscolha uma opção: \n" + "1- Wine\n" + "2- Balance-scale\n" + "3- Exit\n" + "\n>> ")
        if optionDataset in ("1"):
            run("wine", "inicial", 13)
        elif optionDataset in ("2"): 
            run("balance-scale", "inicial", 4)
        elif optionDataset in ("3"):
            break
        else:
            print("Opção inválida") 
    
def run(name, y_position, numberAttributes):

    dataset = Dataset(name, y_position, numberAttributes)
    randomBase = Train_test(dataset)
    randomBase.eighty_by_twenty()

    arvore_1 = Arvore(randomBase)
    arvore_1.treinamento_resultado('entropy')
    
    arvore_2 = Arvore(randomBase)
    arvore_2.treinamento_resultado('gini')

    knN_1 = KnN(randomBase)
    knN_1.treinamento_resultado('euclidean', 5)
    knN_2 = KnN(randomBase)
    knN_2.treinamento_resultado('euclidean', 10)
    knN_3 = KnN(randomBase)
    knN_3.treinamento_resultado('euclidean', 15)
    
    knN_4 = KnN(randomBase)
    knN_4.treinamento_resultado('manhattan', 5)
    knN_5 = KnN(randomBase)
    knN_5.treinamento_resultado('manhattan', 10)
    knN_6 = KnN(randomBase)
    knN_6.treinamento_resultado('manhattan', 15)

    print(f"Arvore_1: {arvore_1}\n")
    print(f"Arvore_2: {arvore_2}\n")

    print(f"KnN_1: {knN_1}\n")
    print(f"KnN_2: {knN_2}\n")
    print(f"KnN_3: {knN_3}\n")
    print(f"KnN_4: {knN_4}\n")
    print(f"KnN_5: {knN_5}\n")
    print(f"KnN_6: {knN_6}\n")

if __name__ == '__main__':
    main()