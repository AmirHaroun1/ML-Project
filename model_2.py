from pre_processing import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def Dec_tree (data) :
    X = data.drop(columns=['PriceRate'])
    Y = data['PriceRate']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),algorithm="SAMME",n_estimators=200)
    bdt.fit(x_train,y_train)

    y_test_pred = bdt.predict(x_test)
    accuracy = np.mean(y_test==y_test_pred)
    print(accuracy)
