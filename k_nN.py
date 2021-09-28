from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class KnN:
    def __init__(self, base, result = None, show = None):
        self._base = base
        self._show = show
        self._result = result

    def treinamento_resultado(self, metrica, n):
        # Treinamento knN
        model = KNeighborsClassifier(n_neighbors=n, metric=f'{metrica}', algorithm='brute')
        model = model.fit(self._base.get_x_train(), self._base.get_y_train())

        # Predição e Resultados

        self._result = model.predict(self._base.get_x_test())
        acc = metrics.accuracy_score(self._result, self._base.get_y_test())
        self._show = round(acc * 100)

    def __str__(self):
        return f"{self._show}% \n {list(self._result)} \n {list(self._base.get_y_test())}"