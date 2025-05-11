from DataProcess import *
from DataVisualize import *
from BinaryClassificationTraining import *
from BinaryClassificationEvaluation import *
from MultiClassificationTraining import *
from MultiClassificationEvaluation import *
import argparse


if __name__ == "__main__":

    hepData = DataProcess("data/HepatitisCdata.csv")
    df = hepData.getFinalData()


    parser = argparse.ArgumentParser(description="Hepatitis C Data Analysis")
    parser.add_argument("--visualize", type=str, help="Visualize data for a specific biomarker (e.g., ALB)")
    parser.add_argument("--train", action="store_true", help="Train binary classification model")
    parser.add_argument("--eval", action="store_true", help="Evaluate binary classification model")
    parser.add_argument("--train_multi", action="store_true", help="Train multi-class classification model")
    parser.add_argument("--eval_multi", action="store_true", help="Evaluate multi-class classification model")

    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        exit()
    # Data Visualization
    if args.visualize:
        if args.visualize == "target":
            DataVisualize(hepData).Target()
        elif args.visualize == "age":
            DataVisualize(hepData).AgeDisease()
        else:
            DataVisualize(hepData).Biomarkers(args.visualize)

    if args.train:
        # Binary Classification Training
        b_train = BinaryClassificationTraining(df)
        b_train.trainit()

    if args.eval:
        # Binary Classification Evaluation
        b_eval = BinaryClassificationEvaluation(df)
        b_eval.evaluate()
        print(b_eval)
        b_eval.ModelsPerformanceBarPlot()
        b_eval.ModelsConfusion()

    # Multi-class Training
    if args.train_multi:
        MultiClassificationTraining(df).trainit()

    # Multi-class Evaluation
    if args.eval_multi:
        m_eval = MultiClassificationEvaluation(df)
        m_eval.evaluate()
        print(m_eval)
        m_eval.ModelsPerformanceBarPlot()
        m_eval.ModelsConfusion()
