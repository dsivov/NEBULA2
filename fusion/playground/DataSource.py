from abc import ABC, abstractmethod
import random

class DataSource:
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_random_pair(self):
        return self[random.randint(0, len(self) - 1)]