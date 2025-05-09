from DataProcess import *
from DataVisualize import *
from BinaryClassificationTraining import *
from BinaryClassificationEvaluation import *
from MultiClassificationTraining import *
from MultiClassificationEvaluation import *



if __name__=="__main__":

    hepData = DataProcess("data/HepatitisCdata.csv")
    df = hepData.getFinalData()

    #DataVisualize(hepData).Biomarkers("ALB")

    ### Binary Case
    #b_train = BinaryClassificationTraining(df)
    #b_eval = BinaryClassificationEvaluation(df)
    #b_eval.evaluate()
    #b_eval.ModelsPerformanceBarPlot()



    ### Multi class Case
    #MultiClassificationTraining(df).trainit()
    m_eval = MultiClassificationEvaluation(df)
    m_eval.evaluate()
    print(m_eval)
    m_eval.ModelsPerformanceBarPlot()
    m_eval.ModelsConfusion()
    
