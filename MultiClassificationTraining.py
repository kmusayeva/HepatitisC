import optuna
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from MultiClassification import MultiClassification
from NNModel import NNModel

# Suppress Optuna logging below WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MultiClassificationTraining(MultiClassification):
    """
    Train and tune multiple multi-class classification models using cross-validation and Optuna.

    Inherits from `MultiClassification`, which is responsible for splitting the input DataFrame
    into `X_train`, `X_test`, `y_train`, `y_test`, and for providing:
      - `self.methods`: List[str] of method names to train
      - `self.cv`: cross-validation splitter
      - `self.mcc_scorer`: scoring callable for Matthews Correlation Coefficient
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the training pipeline and apply SMOTE oversampling.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing features and target column. The parent
            constructor splits into training and test sets.
        """
        super().__init__(df)
        self._apply_smote()

    def _apply_smote(self) -> None:
        """
        Apply SMOTE to balance classes in the training data.

        After this call, `self.X_train` and `self.y_train` are replaced
        by the SMOTE-resampled versions.
        """
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def trainit(self) -> None:
        """
        Train each model specified in `self.methods`, save it to disk, and report progress.

        For each method name, attempts to call `self.{method}_train()`. On success,
        serializes the trained model:
          - PyTorch (`'nn'`): saved with `torch.save`
          - others: pickled via `pickle.dump`

        Raises
        ------
        AttributeError
            If a train function for a method is not defined.
        """
        for method in self.methods:
            print(f">>>> Training model: {method}")
            func_name = f"{method}_train"

            if not hasattr(self, func_name):
                raise AttributeError(f"Training function '{func_name}' not found.")

            train_func = getattr(self, func_name)
            model = train_func()

            # Serialize model
            file_path = f"models/multi/{method}_model.pkl"
            if method == "nn":
                torch.save(model, "models/multi/nn_model.pkl")
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(model, f)

            print(f"Done: {method} saved to {file_path}")

    def rf_train(self) -> RandomForestClassifier:
        """
        Tune and train a RandomForestClassifier via Optuna, optimizing MCC.

        Returns
        -------
        RandomForestClassifier
            The model trained on the entire SMOTE-resampled training set
            with the best-found hyperparameters.
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
            }

            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

            scores = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv, scoring=self.mcc_scorer
            )

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=250)

        print("Best MCC:", round(study.best_value, 2))
        print("Best params:", study.best_params)

        best_clf = RandomForestClassifier(
            **study.best_params, random_state=42, n_jobs=-1
        )
        best_clf.fit(self.X_train, self.y_train)
        return best_clf

    def svc_train(self) -> SVC:
        """
        Tune and train an SVC with RBF kernel via Optuna, optimizing MCC.

        Returns
        -------
        SVC
            The model re-trained on the entire SMOTE-resampled training set
            with the best-found hyperparameters.
        """

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                "kernel": "rbf",
                "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
                "decision_function_shape": "ovr",
            }

            model = SVC(**params, random_state=42, probability=True)

            scores = cross_val_score(
                model, self.X_train, self.y_train, cv=self.cv, scoring=self.mcc_scorer
            )

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150)

        print("Best MCC:", round(study.best_value, 2))
        print("Best params:", study.best_params)

        best_clf = SVC(
            **study.best_params,
            probability=True,
            decision_function_shape="ovr",
            random_state=42,
        )

        best_clf.fit(self.X_train, self.y_train)
        return best_clf

    def nn_train(self) -> torch.nn.Module:
        """
        Tune and train a simple feedforward neural network via Optuna, optimizing MCC.

        Returns
        -------
        torch.nn.Module
            Neural network trained on the full SMOTE-resampled data.
        """

        X_train_tensor = torch.FloatTensor(self.X_train.values)
        y_train_tensor = torch.LongTensor(self.y_train.values)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        num_classes = 4

        def objective(trial):
            hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

            mcc_scores = []

            for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
                train_sub = Subset(train_dataset, train_idx)
                val_sub = Subset(train_dataset, val_idx)
                train_loader = DataLoader(
                    train_sub, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

                model = NNModel(
                    input_dim=self.X_train.shape[1],
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    num_classes=num_classes,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()

                model.train()
                for epoch in range(20):
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()

                model.eval()
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits = model(xb)
                        preds = logits.argmax(dim=1)
                        all_preds.append(preds)
                        all_targets.append(yb)

                y_pred = torch.cat(all_preds).numpy()
                y_true = torch.cat(all_targets).numpy()
                mcc_scores.append(matthews_corrcoef(y_true, y_pred))

            return np.mean(mcc_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150)
        print("Best MCC:", round(study.best_value, 2))
        print("Best params:", study.best_params)

        # Final model training

        best_params = study.best_params

        best_model = NNModel(
            input_dim=self.X_train.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            num_classes=num_classes,
        )

        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])

        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset, batch_size=best_params["batch_size"], shuffle=True
        )

        best_model.train()
        for epoch in range(20):
            for xb, yb in train_loader:
                best_optimizer.zero_grad()
                loss = criterion(best_model(xb), yb)
                loss.backward()
                best_optimizer.step()

        return best_model

    def lr_train(self) -> Pipeline:
        """
        Train a logistic regression with RFECV feature selection.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline containing the RFECV selector and LogisticRegression estimator,
            fitted on the training data.
        """
        lr = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        )
        selector = RFECV(
            estimator=lr,
            step=1,
            cv=self.cv,
            scoring=self.mcc_scorer,
            min_features_to_select=1,
        )
        pipeline = Pipeline([("feature_selection", selector), ("classifier", lr)])

        pipeline.fit(self.X_train, self.y_train)

        return pipeline
