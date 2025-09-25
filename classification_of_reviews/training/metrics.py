import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    weighted_f1 = f1_score(labels, predictions, average="weighted")
    macro_f1 = f1_score(labels, predictions, average="macro")
    accuracy = (predictions == labels).mean()

    return {"weighted_f1": weighted_f1, "macro_f1": macro_f1, "accuracy": accuracy}
