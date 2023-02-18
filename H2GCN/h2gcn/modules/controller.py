from collections import deque


class EarlyStopping:

    def __init__(self, patience=0):
        self._best_value = 0
        self._patience = patience
        self._counter = patience

    @property
    def patience(self):
        return self._patience

    def reset(self):
        pass

    def __call__(self, value):
        if self.patience > 0:
            if value > self._best_value:
                self._best_value = value
                self._counter = self._patience
            else:
                self._counter -= 1
            if self._counter <= 0:
                return True
            return False
        else:
            return False
