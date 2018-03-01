'''Python36
Dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names
'''

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from numpy import sum

def main():
    df = pd.read_csv('iris_flower.txt')
    #df.replace('?', -99999, inplace=True)
        
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
    
    # testing the flower class type with our custom data
    flower_test = np.array([[6.5,2.5,2.9,0.7], [4.2,3.1,5.3,2.6], [5.1,4.6,9.5,1.4]])
    flower_test = flower_test.reshape(len(flower_test), -1)
    predict = clf.predict(flower_test)
    print(predict)
    
    return
#enddef

if __name__ == '__main__':
    main()
#endif