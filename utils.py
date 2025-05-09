import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import optuna
import logging
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict, cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import os
import warnings
from typing import List, Tuple, Optional, Sequence, Any, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def safe_read_csv(filepath, **kwargs):
    """
    Safely reads a CSV file into a pandas DataFrame.

    Parameters:
    - filepath: str, path to the CSV file.
    - **kwargs: Additional arguments passed to pd.read_csv (e.g., delimiter, encoding).

    Returns:
    - df: pandas DataFrame if successful, None otherwise.
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        print(f"File read successfully: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty - {filepath}")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file - {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None


def split_transform(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        drop_alp: bool = False,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and test sets, then apply preprocessing.

    This function performs the following steps:
      1. Stratified train-test split of X and y.
      2. Optionally drops the "ALP" column from both training and test features.
      3. Imputes missing values in numeric columns using k-nearest neighbors.
      4. Standardizes numeric columns to zero mean and unit variance.
      5. Label-encodes the "Sex" column.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix, with both numeric and categorical columns.
    y : pd.Series
        Target vector to predict; used for stratification in the split.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    drop_alp : bool, default=False
        If True, removes the "ALP" column from X_train and X_test before preprocessing.
    random_state: int
        Controls reproducibility of random splitting

    Returns
    -------
    X_train : pd.DataFrame
        Preprocessed feature matrix for training.
    X_test : pd.DataFrame
        Preprocessed feature matrix for testing.
    y_train : pd.Series
        Target vector for the training set.
    y_test : pd.Series
        Target vector for the test set.
    """
    # 1. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 2. Optional column drop
    if drop_alp:
        X_train = X_train.drop(columns="ALP")
        X_test = X_test.drop(columns="ALP")

    # Identify numeric columns
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 3–4. Build and apply numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    num_pipeline.fit(X_train[num_cols])
    X_train[num_cols] = num_pipeline.transform(X_train[num_cols])
    X_test[num_cols] = num_pipeline.transform(X_test[num_cols])

    # 5. Encode binary categorical variable
    encoder = LabelEncoder()
    X_train["Sex"] = encoder.fit_transform(X_train["Sex"])
    X_test["Sex"] = encoder.transform(X_test["Sex"])

    return X_train, X_test, y_train, y_test




def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    method: str,
    pos_label: int = 1
    ) -> pd.Series:
    """
    Compute common binary-classification metrics and return them as a pandas Series.

    This function calculates:
      - Sensitivity (Recall for the positive class)
      - Specificity (Recall for the negative class)
      - F1 score for the positive class
      - Area Under the ROC Curve (AUC)
      - Matthews Correlation Coefficient (MCC)

    Parameters
    ----------
    y_true : Sequence[int]
        True class labels (binary values, e.g. 0 and 1).
    y_pred : Sequence[int]
        Predicted class labels (binary values, matching `y_true`).
    y_prob : Optional[Sequence[float]]
        Predicted probabilities or decision scores for the positive class.
        If `None`, AUC will be returned as NaN.
    method : str
        Name or identifier for this set of predictions.
        Used as the `name` attribute of the returned Series.
    pos_label : int, default=1
        The label in `y_true`/`y_pred` considered the “positive” class.

    Returns
    -------
    metrics : pd.Series
        A pandas Series with index:
          - 'sensitivity'   : float between 0 and 1
          - 'specificity'   : float between 0 and 1
          - 'f1_score'      : float between 0 and 1
          - 'auc'           : float between 0 and 1 (or NaN)
          - 'mcc'           : float between -1 and 1
        Each metric is rounded to two decimal places. The Series is named `method`.
    """
    # Construct confusion matrix [TN, FP, FN, TP]
    labels = [1 - pos_label, pos_label]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    # Calculate sensitivity (true positive rate) and specificity (true negative rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 score and Matthews Correlation Coefficient
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC (if probabilities provided)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')

    # Package and return results
    return pd.Series(
        {
            'sensitivity': round(sensitivity, 2),
            'specificity': round(specificity, 2),
            'f1_score':    round(f1, 2),
            'auc':         round(auc, 2),
            'mcc':         round(mcc, 2),
        },
        name=method
    )


def evaluate_classification_multiclass(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    method: str
) -> pd.Series:
    """
    Compute multiclass classification metrics and return them as a pandas Series.

    This function calculates:
      - Macro-averaged F1 score (unweighted mean of per-class F1 scores)
      - Micro-averaged F1 score (global F1 computed by counting total true positives, false negatives, and false positives)
      - Weighted-averaged F1 score (mean of per-class F1 scores weighted by support)
      - Matthews Correlation Coefficient (a balanced measure even if classes are of very different sizes)

    Parameters
    ----------
    y_true : Sequence[int]
        True class labels for each sample. Labels can be integers or strings that are hashable.
    y_pred : Sequence[int]
        Predicted class labels for each sample, in the same format and order as `y_true`.
    method : str
        Identifier for this set of predictions. Used as the `name` attribute of the returned Series.

    Returns
    -------
    metrics : pd.Series
        A pandas Series with index:
          - 'macro_f1_score'    : float between 0 and 1
          - 'micro_f1_score'    : float between 0 and 1
          - 'weighted_f1_score' : float between 0 and 1
          - 'mcc'               : float between -1 and 1
        Each metric is rounded to two decimal places. The Series is named `method`.
    """
    # Compute F1 scores with different averaging strategies
    f1_macro    = f1_score(y_true, y_pred, average="macro")
    f1_micro    = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Compute Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # Package and return results
    return pd.Series(
        {
            'macro_f1_score':    round(f1_macro, 2),
            'micro_f1_score':    round(f1_micro, 2),
            'weighted_f1_score': round(f1_weighted, 2),
            'mcc':               round(mcc, 2),
        },
        name=method
    )

