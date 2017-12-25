#Dataset can be downloaded from:
#   https://www.kaggle.com/primaryobjects/voicegender

#Method 1 to import the file
#import csv
#f= open("voice.csv", "rb")
#reader = csv.reader(f)
#for row in reader:
#  print row
                                                                                                        

from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit




#sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return expit( x)     #expit(x) is defined as 1/(1+exp(-x))
      
    
    
    

#Method 2 to import the file
#importing dataset
filename='voice.csv'
train=pd.read_csv(filename)

#print train['label'].dtype

#Method 1-storing first 20 columns(column 0-column 19) as training variables and storing 21th column(column name="label") as target variable
X, y = train.iloc[:, 0:-1].values, train.iloc[:, -1].values


                # Or method 2-
#X=train.iloc[:,0:20]
#y1=train.label  #or  y=train.iloc[:, -1] or  y=train.iloc[:, 20]

                
#converting label column into numeric value using LABEL ENCODING 
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y).T

#print X
#print y
#print X.shape
#print y.shape




#Spliting datasets into training dataset(60%) and testing(40%) dataset
X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.40)

#print X_train.shape,X_test.shape,"y_train",y_train.shape,y_test.shape

np.random.seed(20)

syn0 = 2*np.random.random((20,1900)) - 1
#print "Syn0",syn0.shape

epoch=1000
for iter in xrange(epoch):

    #forward propagataion
    
    l0=X_train
    #print "l0",l0.shape
    #print "l0.T",l0.T.shape
    
    l1=nonlin(np.dot(l0,syn0))
    #print "l1",l1.shape

    #how much did we miss?
    l1_error=y_train-l1
    #print "l1",l1_error.shape

    #multiply how much we missed by the
    #slope of the sigmoid at the values in l1

    li_delta=l1_error * nonlin(l1,True)
    #print "li_delta",li_delta.shape
    
    #update weights
    syn0+= np.dot(l0.T,li_delta)
    #print "syn0",syn0.shape

print "Output after training;"
print l1

