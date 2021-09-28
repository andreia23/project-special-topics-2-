from sklearn.model_selection import train_test_split
import pandas as pd

class Train_test:
    def __init__(self, dataset, x_train = None, x_test = None, y_train = None, y_test = None):
        self._X_train = x_train
        self._X_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._dataset = dataset

    def get_x_train(self):
        return self._X_train

    def get_x_test(self):
        return self._X_test

    def get_y_train(self):
        return self._y_train

    def get_y_test(self):
        return self._y_test

    def eighty_by_twenty(self):
        arquivo = open(f"datasets/{self._dataset.getNome()}.data","r+")
    
        dataset = pd.read_csv(arquivo, header=None)

        if(self._dataset.getYposition() == "inicial"):
            index_Y = 0
            index_inicial = 1
            index_final = self._dataset.getLength()
        else:
            index_Y = self._dataset.getLength()
            index_inicial = 0
            index_final = index_Y - 1

        y = dataset[index_Y]
        X = dataset.loc[:,index_inicial:index_final]

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

    def __str__(self):
        return "Train and Test"