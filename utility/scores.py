from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def calculate_scores(all_labels: np.ndarray, all_preds: np.ndarray) -> tuple:
    """
    Calculate various classification metrics.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.

    Returns:
    tuple: A tuple containing accuracy, precision, recall, F1 score, and classification report.
    """

    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average="macro")
    overall_recall = recall_score(all_labels, all_preds, average="macro")
    overall_f1 = f1_score(all_labels, all_preds, average="macro")

    return overall_accuracy, overall_precision, overall_recall, overall_f1


def get_classification_report(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int,
    class_names: List,
    output_dict: bool = True,
) -> dict:
    """
    Generate a classification report.

    Parameters:
    all_labels (np.ndarray): True labels.
    all_preds (np.ndarray): Predicted labels.
    labels (int): Number of unique labels.
    target_name (List): List of target names for the classes.
    output_dir (bool): If True, returns the report as a string.

    Returns:
    str: Classification report as a string.
    """

    report = classification_report(
        all_labels,
        all_preds,
        labels=range(num_classes),
        target_names=class_names,
        output_dict=output_dict,
    )
    return report


def calculate_other_precision_recall_f1(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Last class index
    i = num_classes - 1

    # Extract TP, FP, FN for the last class
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    # Compute modified precision and recall
    mod_precision = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    mod_recall = FP / (FP + TN) if (TP + FN) > 0 else 0.0
    mod_f1 = (
        2 * (mod_precision * mod_recall) / (mod_precision + mod_recall)
        if (mod_precision + mod_recall) > 0
        else 0.0
    )
    return mod_precision, mod_recall, mod_f1
