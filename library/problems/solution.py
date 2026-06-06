from abc import ABC, abstractmethod


class Solution(ABC):
    def __init__(self, repr=None):
        if repr is None:
            repr = self.random_initial_representation()
        self.repr = repr

    def __repr__(self):
        return str(self.repr)

    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass
