from pre_processing import *
import pandas as pd
import model_1
import model_2
import model_3

data = pd.read_csv('House_Data_Classification.csv')
cleaned_data = pre_processing(data)

print("Logistic Regression : ")
model_1.Logistic_Reg(cleaned_data)

print("De Tree: ")
model_2.Dec_tree(cleaned_data)

print("SVM: ")
model_3.SVM(cleaned_data)

