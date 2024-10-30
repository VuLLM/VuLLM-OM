import dataclasses
from dataclasses import dataclass
from typing import List


class Stat:
    def __init__(self, outs=None, loss=0.0, acc={'all_vul': 0, 'TP': 0, 'FP': 0}, labels=None):
        if labels is None:
            labels = []
        if outs is None:
            outs = []
        self.outs = outs
        self.labels = labels
        self.loss = loss
        self.acc = {'all_vul': acc['all_vul'], 'TP': acc['TP'], 'FP': acc['FP']}

    def __add__(self, other):
        self.acc['all_vul'] += other.acc['all_vul']
        self.acc['TP'] += other.acc['TP']
        self.acc['FP'] += other.acc['FP']
        return Stat(self.outs + other.outs, self.loss + other.loss, self.acc, self.labels + other.labels)

    def __str__(self):
        if self.acc['TP'] + self.acc['FP'] == 0 or self.acc['all_vul'] == 0:
            return f"Loss: {round(self.loss, 4)};"
        presicion = self.acc['TP'] / (self.acc['TP'] + self.acc['FP'])
        recall = self.acc['TP'] / self.acc['all_vul']
        if presicion + recall == 0:
            return f"Loss: {round(self.loss, 4)};"
        return f"Loss: {round(self.loss, 4)};"


class Stats:
    def __init__(self, name: str, results=None, total=None):
        self.name = name
        self.results = results if results is not None else []
        self.total = total if total is not None else Stat()

    def __call__(self, stat):
        self.total = self.total.__add__(stat)
        self.results.append(stat)

    def __str__(self):
        return f"{self.name} {self.mean()}"

    def __len__(self):
        return len(self.results)

    def mean(self):
        res = Stat()
        res = res.__add__(self.total)
        res.loss /= len(self)
        # res.acc /= len(self)

        return res

    def loss(self):
        return self.mean().loss

    def acc(self):
        return self.acc

    def outs(self):
        return self.total.outs

    def labels(self):
        return self.total.labels
