import random

from abc import ABC, abstractmethod


class Sorter(ABC):
    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NoSorter(Sorter):
    def __call__(self, data):
        return data

    def __str__(self):
        return "none"


class RandomSorter(Sorter):
    def __call__(self, data):
        return random.sample(data, len(data))

    def __str__(self):
        return "random"


class KeySorter(Sorter):
    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def __call__(self, data):
        return sorted(data, key=lambda x: x[self.key], reverse=self.reverse)

    def __str__(self):
        return "{}/{}".format(self.key, "desc" if self.reverse else "asc")


# class Filter(ABC):
#     @abstractmethod
#     def __call__(self, data):
#         pass

#     @abstractmethod
#     def __str__(self):
#         pass


# class FilterByKey(Filter):
#     def __init__(self, key, value):
#         self.key = key
#         self.value = value

#     def __call__(self, data):
#         return [item for item in data if item[self.key] == self.value]

#     def __str__(self):
#         return "{}={}".format(self.key, self.value)

