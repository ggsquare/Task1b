import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

trainfilename = '/Users/Anna/polybox/IntroductionML/Tasks/01/task1b_ow9d8s/train.csv'

trainfile = pd.read_csv(trainfilename, delimiter = ',')

X = trainfile._drop_axis(['Id','y'], axis=1)
y = trainfile['y']

X=np.array(X) #900x5
y=np.array(y)

x1=X[:,0]
x2=X[:,1]
x3=X[:,2]
x4=X[:,3]
x5=X[:,4]

phi=np.transpose([x1, x2, x3, x4, x5, x1**2, x2**2, x3**2, x4**2, x5**2, np.exp(x1), np.exp(x2), np.exp(x3), np.exp(x4), np.exp(x5), np.cos(x1), np.cos(x2), np.cos(x3), np.cos(x4), np.cos(x5), np.ones(900)]) #900X21
phi2=X #900X5
phi2=np.append(phi2,X**2,axis=1) #900X10
phi2=np.append(phi2, np.exp(X), axis=1) #900x15
phi2=np.append(phi2, np.cos(X), axis=1) #900X20
phi2=np.append(phi2, np.ones((900,1)), axis=1) #900x21

#we need to split into a test set and a train set
#to do so we could use eg. k-fold corss validation --> often k=5 or k=10
kf = KFold(n_splits=10)
regr = linear_model.LinearRegression()
#regr = linear_model.Lasso(alpha=1)
#regr=Ridge(alpha=0.1)
RMSE=[]
possibleWeight =[]

for train_index, test_index in kf.split(phi):
    weights=[]
    X_train, X_test = phi[train_index], phi[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)

    RMSE.append(mean_squared_error(y_pred, y_test)**0.5)
    weights=regr.coef_
    possibleWeight.append(weights)

print(RMSE)
print(np.min(RMSE))
indxminRMSE=np.argmin(RMSE)
print(indxminRMSE)
# output results
d={'weight': possibleWeight[indxminRMSE]}
output=pd.DataFrame(d)
output.to_csv('task_1b_output.csv', index=False, header=False)


#Insights: RMSE with linear regression too big (?) for all folds! minimum is at 9.716, index=4
#RMSE with Ridge regression, alpha = 0.1 -> min(RMSE)=9.7154
#RMSE wiht Lasse regression, alpha = 0.1 -->min(RMSE)=9.686
