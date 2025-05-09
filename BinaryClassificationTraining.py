from utils import *
from BinaryClassification import *
from NNModel import NNModel


# Suppress Optuna logging below WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

class BinaryClassificationTraining(BinaryClassification):
    """
    Train and tune multiple binary classification models using cross-validation and Optuna.

    Inherits from `Classification`, which is responsible for splitting the input DataFrame
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
            file_path = f"models/binary/{method}_model.pkl"
            if method == "nn":
                torch.save(model, "models/binary/nn_model.pkl")
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
        def objective(trial: optuna.Trial) -> float:
            params: Dict[str, Any] = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            scores = cross_val_score(
                clf, self.X_train, self.y_train,
                cv=self.cv, scoring=self.mcc_scorer
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=250)
        print("Best MCC:", study.best_value)
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
        def objective(trial: optuna.Trial) -> float:
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'kernel': 'rbf',
                'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True)
            }
            clf = SVC(**params, probability=True, random_state=42)
            scores = cross_val_score(
                clf, self.X_train, self.y_train,
                cv=self.cv, scoring=self.mcc_scorer
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150)
        print("Best MCC:", study.best_value)
        print("Best params:", study.best_params)

        best_clf = SVC(
            **study.best_params, probability=True, random_state=42
        )
        best_clf.fit(self.X_train, self.y_train)
        return best_clf

    def xgb_train(self) -> xgb.XGBClassifier:
        """
        Tune and train an XGBoost classifier via Optuna, optimizing MCC.

        Returns
        -------
        xgb.XGBClassifier
            The model re-trained on the entire SMOTE-resampled training set
            with the best-found hyperparameters.
        """
        def objective(trial: optuna.Trial) -> float:
            param: Dict[str, Any] = {
                'verbosity': 0,
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0)
            }
            if param['booster'] in ['gbtree', 'dart']:
                param.update({
                    'max_depth': trial.suggest_int('max_depth', 3, 9, step=2),
                    'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
                    'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                })
            if param['booster'] == 'dart':
                param.update({
                    'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
                    'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
                    'rate_drop': trial.suggest_float('rate_drop', 1e-8, 1.0, log=True),
                    'skip_drop': trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
                })
            # CV with manual XGBoost loop
            mccs = []
            for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
                booster = xgb.train(
                    param, dtrain, num_boost_round=100,
                    evals=[(dval, 'validation')], early_stopping_rounds=10,
                    verbose_eval=False
                )
                preds = (booster.predict(dval) > 0.5).astype(int)
                mccs.append(matthews_corrcoef(y_val, preds))  # type: ignore
            return float(np.mean(mccs))  # type: ignore

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=250, timeout=600)
        print("Best MCC:", study.best_value)
        print("Best params:", study.best_params)

        final_model = xgb.XGBClassifier(
            **study.best_params, enable_categorical=True
        )
        final_model.fit(self.X_train, self.y_train)
        return final_model

    def nn_train(self) -> torch.nn.Module:
        """
        Tune and train a simple feedforward neural network via Optuna, optimizing MCC.

        Returns
        -------
        torch.nn.Module
            Neural network trained on the full SMOTE-resampled data.
        """
        # Prepare PyTorch dataset
        X_tensor = torch.FloatTensor(self.X_train.values)
        y_tensor = torch.FloatTensor(self.y_train.values).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'hidden_dim': trial.suggest_int('hidden_dim', 16, 128),
                'n_layers': trial.suggest_int('n_layers', 1, 3),
                'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            mccs: list[float] = []
            for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
                train_sub = Subset(dataset, train_idx)
                val_sub = Subset(dataset, val_idx)
                train_loader = DataLoader(train_sub, batch_size=params['batch_size'], shuffle=True)
                val_loader   = DataLoader(val_sub,   batch_size=params['batch_size'], shuffle=False)

                model = NNModel(
                    input_dim=self.X_train.shape[1],
                    hidden_dim=params['hidden_dim'],
                    n_layers=params['n_layers']
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                criterion = torch.nn.BCEWithLogitsLoss()

                # Train loop
                model.train()
                for _ in range(20):
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(model(xb), yb)
                        loss.backward()
                        optimizer.step()

                # Validation
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        out = torch.sigmoid(model(xb))
                        out = (out > 0.5).float()
                        preds.append(out)
                        targets.append(yb)
                preds = torch.cat(preds).numpy().ravel()
                targets = torch.cat(targets).numpy().ravel()
                mccs.append(matthews_corrcoef(targets, preds))  # type: ignore

            return np.mean(mccs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150)
        print("Best MCC:", study.best_value)
        print("Best params:", study.best_params)

        # Final model training
        best = study.best_params
        final_model = NNModel(
            input_dim=self.X_train.shape[1],
            hidden_dim=best['hidden_dim'],
            n_layers=best['n_layers']
        )
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
        criterion = torch.nn.BCEWithLogitsLoss()
        train_loader = DataLoader(dataset, batch_size=best['batch_size'], shuffle=True)
        final_model.train()
        for _ in range(20):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                criterion(final_model(xb), yb).backward()
                optimizer.step()
        return final_model

    def lr_train(self) -> Pipeline:
        """
        Train a logistic regression with RFECV feature selection.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline containing the RFECV selector and LogisticRegression estimator,
            fitted on the training data.
        """
        lr = LogisticRegression(penalty='l2',
                                solver='liblinear',
                                max_iter=1000,
                                random_state=42
                                )
        selector = RFECV(
            estimator=lr,
            step=1,
            cv=self.cv,
            scoring=self.mcc_scorer,
            min_features_to_select=1
        )
        pipeline = Pipeline([('feature_selection', selector), ('classifier', lr)])
        pipeline.fit(self.X_train, self.y_train)
        return pipeline
