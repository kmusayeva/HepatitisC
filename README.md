# Description

This project represents statistical and predictive modeling done on the Hepatitis C dataset publicly available at UCI website.

# Data
The dataset consists of 615 patients characterized by age, sex, and laboratory tests values of 10 biochemical blood test markers. The target consists of 4 classes: Blood Donor, Hepatitis-C, Fibrosis, Cirrhosis. Only 12% of the dataset is disease cases. 

Statistical analysis computes the prevalence of disease cases across variables, and adjusted odds ratio analysis.


# Predictive Modeling

Logistic regression, support vector classifier, feed-forward neural networks are trained on the random half of the data, where the disease cases are artificially increased thanks to SMOTE approach.

# Evaluation Measures

F1, AUC, sensitivity, specificity, Matthews correlations coefficients are used as evaluation measures in the binary classification setting, and macro and micro F1 measures and Matthews correlation coefficient are used as evaluation measures in the multi-class classification setting. In both setting, the models are trained based on the Matthews correlation coefficient which evaluates the confusion matrix in its entirety.

# Results

In the binary setting, logistic regression has a higher sensitivity, and random forest and neural networks have higher and similar specificities.

In the multi-class classification setting,  logistic regression makes fewer errors in predicting hepatitis C and fibrosis cases, and equally well as neural network in predicting cirrhosis cases. However, it misclassifies many blood donors as disease cases.

It should be noted that quantitave models like logistic regression is supported by *Arden Syntax* which programs medical logic for clinical decision support, but not black box models of higher capacity such as random forests or neural networks.


