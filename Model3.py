from sklearn import svm
from sklearn.model_selection import train_test_split

from pre_processing import *
import pandas as pd


def Model3(cleaned_data) :
    corr = cleaned_data.corr()
    c_top_features = corr.index[abs(corr["PriceRate"]) > 0.5]
    c_top_features = c_top_features.delete(-1)

    Y = cleaned_data["PriceRate"]
    X = cleaned_data.drop(columns = ["PriceRate"])

    X = X.loc[:,c_top_features]


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=None, shuffle=True)

    SVcmodel = svm.SVC(kernel='poly',degree=2, C=2)
    SVcmodel.fit(x_train, y_train)
    predictions1 = SVcmodel.predict(x_test)
    # predictions2 = SVcmodel.predict(x_train)
    accuracy1 = np.mean(predictions1 == y_test)
    # accuracy2 = np.mean(predictions2 == y_train)

    #MSE
    MSE = metrics.mean_squared_error(y_test, predictions1)
    print("MSE : ", MSE)

    #r_2_score
    r_2_score = metrics.r2_score(y_test, predictions1)
    print("r_2_score : ",r_2_score)

    #Accuracy
    accuracy = np.mean(y_test == predictions1)
    print("Accuracy : ",accuracy)


data = pd.read_csv('House_Data_Classification.csv')
cleaned_data = pre_processing(data)
Model3(cleaned_data)