import numpy as np
import pandas as pd
from utils import safe_read_csv


class DataProcess:

    def __init__(self, filename):
        self.df = safe_read_csv(filename)
        self.diagnosis_dict = {
            "Blood Donor": 0,
            "Hepatitis": 1,
            "Fibrosis": 2,
            "Cirrhosis": 3,
        }
        self.clean()
        self.createLabels()

    def printData(self):
        print(self.df.head(15))

    def clean(self):
        self.df.drop(columns=["Unnamed: 0"], inplace=True)
        self.df.rename(columns={"Category": "Diagnosis"}, inplace=True)
        self.df.replace("0s=suspect Blood Donor", "Blood Donor", inplace=True)
        self.df.replace("0=Blood Donor", "Blood Donor", inplace=True)
        self.df.replace("1=Hepatitis", "Hepatitis", inplace=True)
        self.df.replace("2=Fibrosis", "Fibrosis", inplace=True)
        self.df.replace("3=Cirrhosis", "Cirrhosis", inplace=True)

    def createLabels(self):
        self.df["Diagnosis_Multi"] = self.df["Diagnosis"].map(self.diagnosis_dict)
        self.df["Diagnosis_Binary"] = self.df["Diagnosis_Multi"].apply(
            lambda x: 1 if x > 0 else 0
        )

    def getFinalData(self):
        return self.df
