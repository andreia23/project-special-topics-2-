class Dataset:
    def __init__(self, nome, y_position , length):
        self._nome = nome
        self._y_position = y_position
        self._length = length

    def getNome(self):
        return self._nome
        
    def getYposition(self):
        return self._y_position

    def getLength(self):
        return self._length
    
    def __str__(self):
        return f"\n {self._nome} \n {self._y_position} \n {self._length}"