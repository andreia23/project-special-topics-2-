from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter

class Kmean:
    def __init__(self, base, result = None, show = None):
        self._base = base
        self._show = show
        self._result = result

    def treinamento_resultado(self):
        # Treinamendo do kmeans
        myset = set(self._base.get_y_train())
        clusters = len(myset)
        model = KMeans(n_clusters = clusters)
        model = model.fit(self._base.get_x_train())
        
        # Executar mapeamento das classes aos clusters

        # Pegar os labels dos padrões de Treinamento
        labels = model.labels_
        
        map_labels = []

        for i in range(clusters):
            map_labels.append([])

        new_y_train = self._base.get_y_train().to_list()

        for i in range(len(self._base.get_y_train())):
            for c in range(clusters):
                if labels[i] == c:
                    map_labels[c].append(new_y_train[i])

        # Criar dicionário com os labells a serem mapeados
        mapping = {}

        for i in range(clusters):
            final = Counter(map_labels[i]) # contar a classe que mais aparece
            value = final.most_common(1)[0][0] # retorna a classe com maior frequência
            mapping[i] = value

        # Predição e Resultados

        self._result = model.predict(self._base.get_x_test())
        self._result = [mapping[i] for i in self._result]
        acc = metrics.accuracy_score(self._result, self._base.get_y_test())
        self._show = round(acc * 100)

    def __str__(self):
        return f"{self._show}% \n {list(self._result)} \n {list(self._base.get_y_test())}"