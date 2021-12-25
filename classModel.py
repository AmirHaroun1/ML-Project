from pre_processing import *
import pandas as pd


data = pd.read_csv('House_Data_Classification.csv')
cleaned_data = pre_processing(data)
print (pd.DataFrame(cleaned_data).info())

