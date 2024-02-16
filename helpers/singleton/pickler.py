import pickle
from pathlib import Path

from helpers.singleton.base import SingletonMeta


class PickleOperator(metaclass=SingletonMeta):
    def dump(self, filename: str|Path, obj):
        with open(Path(filename), 'wb') as pickle_file:
            pickle.dump(obj, pickle_file)

    def load(self, filename: str|Path):
        if not Path(filename).exists():
            return
        with open(filename, 'rb') as pickle_file:
            return pickle.load(pickle_file)