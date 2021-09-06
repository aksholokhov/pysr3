from typing import Set


class Logger:
    """
    Helper class for logging the progress of iterative methods.
    """
    def __init__(self, list_of_keys: Set = ()):
        """
        Initializes the logger

        Parameters
        ----------
        list_of_keys: set[str]
            list of keys for the logger
        """
        self.keys = list_of_keys
        self.dict = {key: [] for key in list_of_keys}

    def log(self, parameters):
        """
        Records all values of parameters which keys are already in the logger.
        Ignores the rest.

        Parameters
        ----------
        parameters: dict
            dictionary with parameters to record.

        Returns
        -------
        self
        """
        for key in self.keys:
            self.dict[key].append(parameters.get(key, None))
        return self

    def add(self, key, value):
        """
        Adds a key-value pair to the logger

        Parameters
        ----------
        key: str
            key
        value: Any
            value for this key

        Returns
        -------

        """
        self.dict[key] = value
        return self

    def append(self, key, value):
        """
        Adds value to what's already stored in the logger. If no such key then it starts with 0.

        Parameters
        ----------
        key: str
            key
        value: Any additive
            value to add

        Returns
        -------
        self
        """
        self.dict[key] += value
        return self

    def get(self, key):
        """
        Returns the value by key

        Parameters
        ----------
        key: str
            key

        Returns
        -------
        value for this key
        """
        return self.dict[key]
