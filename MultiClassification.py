from utils import *


class MultiClassification:

    def __init__(self, df):

        self.X = df.drop(columns=["Diagnosis", "Diagnosis_Binary", "Diagnosis_Multi"])

        self.y = df["Diagnosis_Multi"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            split_transform(self.X, self.y, 0.5, True, random_state=42))

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.mcc_scorer = make_scorer(matthews_corrcoef)

        self.methods = ["svc", "rf",  "nn", "lr"]


