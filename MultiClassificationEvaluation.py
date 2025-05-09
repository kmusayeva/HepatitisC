from utils import *
from MultiClassification import MultiClassification



class MultiClassificationEvaluation(MultiClassification):
    """
    Evaluate multiple binary classification models on a held‐out test set.

    Inherits from `Classification`, which is assumed to split the data into
    train and test sets.

    Attributes
    ----------
    __results : pd.DataFrame, aggregated metrics for each classifier after calling `evaluate()`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the evaluator with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset containing both features and the target column.
            The parent `Classification` constructor will handle splitting.
            Reproducibility is controlled by random_state.
        """
        super().__init__(df)
        self.preds_dict: Dict = {}
        self.__results: pd.DataFrame = None

    def evaluate(self) -> None:
        """
        Load each specified model, generate predictions, compute metrics, and store them.

        After predictions, calls `evaluate_classification` to compute:
          macro, micro, weighted F1 and MCC measures

        Stores the concatenated pandas DataFrame of all results in `self.__results`,
        with one column per method.
        """
        series_list: List[pd.Series] = []

        for method in self.methods:
            if method == "nn":
                # Suppress future warnings from PyTorch
                warnings.filterwarnings("ignore", category=FutureWarning)
                model: torch.nn.Module = torch.load("models/multi/nn_model.pkl", weights_only=False)
                model.eval()

                # Prepare test tensor and get preds
                X_test_tensor = torch.FloatTensor(self.X_test.values)
                with torch.no_grad():
                    preds = model(X_test_tensor).argmax(dim=1).numpy()
                    

            else:
                # Load a scikit-learn model and get predictions & probabilities
                path = f"models/multi/{method}_model.pkl"
                with open(path, 'rb') as f:
                    model = pickle.load(f)

                preds = model.predict(self.X_test)

            
            # Compute metrics for this method
            metrics_series = evaluate_classification_multiclass(
                y_true=self.y_test,
                y_pred=preds,
                method=method
            )
            series_list.append(metrics_series)
            self.preds_dict[method] = preds
        # Combine into a DataFrame: columns are method names, rows are metric names
        self.__results = pd.concat(series_list, axis=1)




    def ModelsPerformanceBarPlot(self) -> None:

        """
        Plot the performance metrics of all methods in a bar plot.

        The plot includes:
          - Sensitivity
          - Specificity
          - F1 score
          - AUC
          - Matthews Correlation Coefficient (MCC)
        
        """

        fig, ax = plt.subplots(figsize=(17, 5))
        self.__results.plot(kind='bar', ax=ax)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Methods Across Metrics')
        ax.legend(title='Method')       
        ax.grid(False)
        plt.xticks(rotation=0)          
        plt.tight_layout()
        plt.show()



    def ModelsConfusion(self) -> None:
        """
        Plot a confusion matrix of all methods.
        """

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        for i, (method, preds) in enumerate(self.preds_dict.items()):

                cm = confusion_matrix(self.y_test, preds)

                class_names = ['No Disease', 'Hepatitis C', 'Fibrosis', 'Cirrhosis']

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

                disp.plot(cmap='Blues', values_format='d', ax=axes[i], colorbar=False, text_kw={'fontsize':16})

                axes[i].grid(False)

                axes[i].set_ylabel('')
                axes[i].set_xlabel('Predicted Label', fontsize=14)

                axes[i].set_title(method)

        plt.tight_layout()
        plt.show()



    def __str__(self) -> str:
        """
        Retrieve the DataFrame of evaluation metrics.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column corresponds to one model/method and
            each row is a metric (sensitivity, specificity, F1, AUC, MCC).
            Returns None if `evaluate()` has not been run yet.
        """
        return f"{self.__results}"
