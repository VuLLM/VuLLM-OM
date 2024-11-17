import torch
from ..utils.objects import stats


def f1_score(probs, all_labels):
    # Convert probabilities and labels to binary predictions and actual labels
    predictions = [1 if float(p) >= 0.5 else 0 for p in probs]
    actuals = [int(l) for l in all_labels]

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
    false_positives = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))

    # Avoid division by zero
    return {'all_vul': sum(actuals), 'TP': true_positives, 'FP': false_positives}


class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer

    def __call__(self, i, x, y):
        out = self.model(x)
        loss = self.criterion(out, y.float())
        acc = f1_score(out, y.float())

        if self.model.training:
            # calculates the gradient
            loss.backward()
            # and performs a parameter update based on it
            self.optimizer.step()
            # clears old gradients from the last step
            self.optimizer.zero_grad()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(out.tolist(), loss.item(), acc, y.tolist())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
