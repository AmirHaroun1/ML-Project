from pre_processing import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def Dec_tree (df) :
    X = df.drop(columns=['PriceRate'])
    Y = df['PriceRate']
    # Features That Correlation is higher than average
    corr = df.corr()
    c_top_features = corr.index[abs(corr['PriceRate']) > 0.5]
    c_top_features = c_top_features.delete(-1)
    #X = X.loc[:, c_top_features]

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),algorithm="SAMME",n_estimators=200)
    bdt.fit(x_train,y_train)

    y_pred = bdt.predict(x_test)
    accuracy = np.mean(y_test==y_pred)
    # MSE
    MSE = metrics.mean_squared_error(y_test, y_pred)
    print("MSE : ", MSE)

    # r_2_score
    r_2_score = metrics.r2_score(y_test, y_pred)
    print("r_2_score : ", r_2_score)

    # Accuracy
    accuracy = np.mean(y_test == y_pred)
    print("Accuracy : ", accuracy)
