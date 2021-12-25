import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def Logistic_Reg(df) :
    X = df[df.column != 'PriceRate']
    Y = df['PriceRate']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

    logisticRegr = LogisticRegression()

    # FIT Train Data
    logisticRegr.fit(x_train, y_train)
    # Predict Test Data
    y_pred = logisticRegr.predict(x_test)

    #MSE
    MSE = metrics.mean_squared_error(y_test, y_pred)
    print("MSE : ", MSE)

    #r_2_score
    r_2_score = metrics.r2_score(y_test, y_pred)
    print("r_2_score : ",r_2_score)

    #Accuracy
    accuracy = np.mean(y_test == y_pred)
    print("Accuracy : ",accuracy)
