'''Python36
Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
import random

def main():
    df = pd.read_csv('wine.txt')
    X = np.array(df.drop(['class'], 1))
    Y = np.array(df['class'])
    
    count = 0
    accuracy = []
    while count<=10:
        
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
        
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, Y_train)
        
        accuracy.append(clf.score(X_test, Y_test))
        
        count+=1
    #endwhile
    
    avg_accuracy = sum(np.array(accuracy))/len(accuracy)
    print(avg_accuracy)
    
    # testing the wine class type with our custom data
    wine_test = np.array([[10.6,2,2.61,17.6,11,2.6,5.51,.31,1.25,5.05,1.06,3.58,1095], [11.82,1.72,1.88,19.5,86,2.5,1.64,.37,1.42,2.06,.94,2.44,415]])
    wine_test = wine_test.reshape(len(wine_test), -1)
    predict = clf.predict(wine_test)
    print(predict)
    
    return
#enddef

if __name__ == '__main__':
    main()
#endif