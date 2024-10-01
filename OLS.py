import pandas as pd
import numpy as np

df = pd.read_csv('Salary_dataset.csv')
df = df.iloc[:,1:]
X = df.iloc[:,0].values
y= df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state= 1)

class SelfOLR :
    def __init__ (self):
        self.coef_ =0
        self.intercept_ =0

    def fit(self, X_train, y_train):
        x_mean= np.mean(X_train)
        y_mean = np.mean(y_train)
        num = 0
        den= 0
        for i in range(len(X_train)):
            num = num + (X_train[i]-x_mean)*(y_train[i]-y_mean)
            den = den+ ((X_train[i]-x_mean)**2)

        self.coef_  = num/den
        self.intercept_ = y_mean - self.coef_*x_mean

    def predict (self, X_test) :
        return X_test * self.coef_ + self.intercept_

reg = SelfOLR()
reg.fit(X_train,y_train)