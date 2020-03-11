from typing import Set


class Logger:
    def __init__(self, list_of_keys: Set = ()):
        self.keys = list_of_keys
        self.dict = {key: [] for key in list_of_keys}

    def log(self, parameters):
        for key in self.keys:
            self.dict[key].append(parameters.get(key, None))
        return self

    def add(self, key, value):
        self.dict[key] = value
        return self

    def get(self, key):
        return self.dict[key]
