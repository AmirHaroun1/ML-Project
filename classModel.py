from pre_processing import *
import pandas as pd
import model_1
import model_2

data = pd.read_csv('House_Data_Classification.csv')
cleaned_data = pre_processing(data)
print("Logistic Regression : ")
model_1.Logistic_Reg(cleaned_data)
print("Decesion Tree: ")
model_2.Dec_tree(cleaned_data)