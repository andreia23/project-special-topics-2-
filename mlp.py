from sklearn.neural_network import MLPClassifier
from sklearn import metrics

class Mlp:
    def __init__(self, base, result = None, show = None):
        self._base = base
        self._show = show
        self._result = result

    def treinamento_resultado(self, neuroniosPorCamada):
        # Treinamendo do Mlp    
        model = MLPClassifier(hidden_layer_sizes= neuroniosPorCamada, activation='tanh', max_iter=2000)
        model = model.fit(self._base.get_x_train(), self._base.get_y_train())

        # Predição e Resultados

        self._result = model.predict(self._base.get_x_test())
        acc = metrics.accuracy_score(self._result, self._base.get_y_test())
        self._show = round(acc * 100)

    def __str__(self):
        return f"{self._show}% \n {list(self._result)} \n {list(self._base.get_y_test())}"