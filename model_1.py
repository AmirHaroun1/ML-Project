import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import os
def Logistic_Reg(df) :

    X = df.drop(columns=['PriceRate'])
    Y = df['PriceRate']

    # Features That Correlation is higher than average
    corr = df.corr()
    c_top_features = corr.index[abs(corr['PriceRate']) > 0.5]
    c_top_features = c_top_features.delete(-1)
    X = X.loc[:,c_top_features]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0,shuffle=True)

    logisticRegr = pickle.load(open('LogisticReg.sav', 'rb'))

    # Predict Test Data
    y_pred = logisticRegr.predict(x_test)

    #MSE
    MSE = metrics.mean_squared_error(y_test, y_pred)
    print("\tMSE : ", MSE)

    #r_2_score
    r_2_score = metrics.r2_score(y_test, y_pred)
    print("\tr_2_score : ",r_2_score)

    #Accuracy
    accuracy = np.mean(y_test == y_pred)
    print("\tAccuracy : ",accuracy)
