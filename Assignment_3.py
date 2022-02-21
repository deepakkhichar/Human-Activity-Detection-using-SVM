#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix


# # Loading Dataset (Train and Test Data)

# In[2]:


filename = 'train.csv'
trainData = pd.read_csv(filename)
print(trainData.shape)
testData = pd.read_csv('test.csv')
print(testData.shape)


# In[3]:


trainData.head()


# # Preprocessing

# #### Checking for Duplicates

# In[4]:


print('No of duplicates in Train Data:',sum(trainData.duplicated()))
print('No of duplicates in Test Data :',sum(testData.duplicated()))


# #### Checking for NaN/null values

# In[5]:



print('No of NaN/Null values in Train Data:',trainData.isnull().values.sum())
print('No of NaN/Null values in Test Data:',testData.isnull().values.sum())


# #### Data prepration
# 

# In[6]:


X_train = trainData.drop(['Activity'], axis = 1)
Y_train = trainData['Activity']
X_test = testData.drop(['Activity'], axis = 1)
Y_test = testData['Activity']

# Transforming non-numerical value in Y to numerical value using label Encoder
le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
Y_test = le.transform(Y_test)
print(le.classes_)


# #### Normalising the data

# In[7]:


from sklearn import preprocessing
X_train_n = preprocessing.normalize(X_train)
X_test_n =  preprocessing.normalize(X_test)


# #### Feature Scaling 

# In[8]:


# MinMaxScalar
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler().fit(X_train_n)
X_train_mm = scaler1.transform(X_train_n)
X_test_mm = scaler1.transform(X_test_n)

# Standard Scaler
# from sklearn.preprocessing import StandardScaler
# scaler2 = StandardScaler().fit(X_train_n)
# X_train_st = scaler2.transform(X_train_n)
# X_test_st = scaler2.transform(X_test_n)


# #### Preparing training and validation sets

# In[9]:


from sklearn.model_selection import train_test_split
X_training, X_validation, Y_training, Y_validation = train_test_split(X_train_mm, Y_train, test_size=0.2, random_state=42)
X_testing=X_test_mm
Y_testing=Y_test


# # Training the model

# ## SVM with Linear kernel

# In[10]:


svc_linear = SVC(kernel='linear',C=100) 
svc_linear.fit(X_training,Y_training)


Y_training_pred = svc_linear.predict(X_training)
Y_validation_pred = svc_linear.predict(X_validation)
Y_testing_pred = svc_linear.predict(X_testing)

print("SVM with Linear Kernel")
print('----------------------------------------------------------------------------------------------------')
print()
print('Tuned parameter Values, C :',svc_linear.C)
print()
print()
print('Accuracy Score on Training Data:',accuracy_score(Y_training, Y_training_pred))
print()
print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Validation Set--------------------------------------------')
print()
print('Accuracy Score on Validation Data:',accuracy_score(Y_validation, Y_validation_pred))
print()
#CM=confusion_matrix(Y_validation, Y_validation_pred)
print("Confusion Matrix on Validation Data:")
#print(CM)
plot_confusion_matrix(svc_linear, X_validation, Y_validation)  
plt.show()
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
print("Normalised Confusion Matrix on Validation Data:")
#print(CM_normalised)
plot_confusion_matrix(svc_linear, X_validation, Y_validation,normalize='true')  
plt.show()
print("Classification Report on Validation Data:")
print(classification_report(Y_validation,Y_validation_pred))
print()


print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Testing Set-----------------------------------------------')
print()
print('Accuracy Score on Testing Data:',accuracy_score(Y_testing, Y_testing_pred))
print()
print("Confusion Matrix on Test Data:")
#CM=confusion_matrix(Y_test, Y_test_pred)
#print(CM)
plot_confusion_matrix(svc_linear, X_test_mm, Y_test)  
plt.show()
print("Normalised Confusion Matrix on Test Data:")
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
#print(CM_normalised)
plot_confusion_matrix(svc_linear, X_test_mm, Y_test,normalize='true')  
plt.show()
print("Classification Report on Test Data:")
print(classification_report(Y_testing, Y_testing_pred))


# In[11]:


C_trials = [0.001, 0.01, 0.1, 1, 10, 100,1000]

for i in range(len(C_trials)):
    svc_linear = SVC(kernel='linear',C=C_trials[i]) 
    svc_linear.fit(X_training,Y_training)


    Y_training_pred = svc_linear.predict(X_training)
    Y_validation_pred = svc_linear.predict(X_validation)
    Y_testing_pred = svc_linear.predict(X_testing)



    print('Accuracy Score on Testing Data with C as', str(C_trials[i]),' :',accuracy_score(Y_testing, Y_testing_pred))


# ## SVM with RBF kernel 

# In[12]:



svc_rbf = SVC(kernel='rbf',gamma=0.02,C=100)  # gamma='scale'
svc_rbf.fit(X_training,Y_training)


Y_training_pred = svc_rbf.predict(X_training)
Y_validation_pred = svc_rbf.predict(X_validation)
Y_testing_pred = svc_rbf.predict(X_testing)

print("SVM with RBF Kernel")
print('----------------------------------------------------------------------------------------------------')
print()
print('Tuned parameter Values, C :',svc_rbf.C,', gamma :',svc_rbf._gamma)
print()
print()
print('Accuracy Score on Training Data:',accuracy_score(Y_training, Y_training_pred))
print()
print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Validation Set--------------------------------------------')
print()
print('Accuracy Score on Validation Data:',accuracy_score(Y_validation, Y_validation_pred))
print()
#CM=confusion_matrix(Y_validation, Y_validation_pred)
print("Confusion Matrix on Validation Data:")
#print(CM)
plot_confusion_matrix(svc_rbf, X_validation, Y_validation)  
plt.show()
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
print("Normalised Confusion Matrix on Validation Data:")
#print(CM_normalised)
plot_confusion_matrix(svc_rbf, X_validation, Y_validation,normalize='true')  
plt.show()
print("Classification Report on Validation Data:")
print(classification_report(Y_validation,Y_validation_pred))
print()


print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Testing Set-----------------------------------------------')
print()
print('Accuracy Score on Testing Data:',accuracy_score(Y_testing, Y_testing_pred))
print()
print("Confusion Matrix on Test Data:")
#CM=confusion_matrix(Y_test, Y_test_pred)
#print(CM)
plot_confusion_matrix(svc_rbf, X_test_mm, Y_test)  
plt.show()
print("Normalised Confusion Matrix on Test Data:")
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
#print(CM_normalised)
plot_confusion_matrix(svc_rbf, X_test_mm, Y_test,normalize='true')  
plt.show()
print("Classification Report on Test Data:")
print(classification_report(Y_testing, Y_testing_pred))


# In[13]:


trials = [0.001, 0.01, 0.1, 1, 10, 100]
#Make grid using above list
grid = []
for i in range(6):
    for j in range(4):
        grid.append((trials[i],trials[j]))

for i in range(len(grid)):
    svc_rbf = SVC(kernel='rbf',gamma=grid[i][1],C=grid[i][0])  # gamma='scale'
    svc_rbf.fit(X_training,Y_training)


    Y_training_pred = svc_rbf.predict(X_training)
    Y_validation_pred = svc_rbf.predict(X_validation)
    Y_testing_pred = svc_rbf.predict(X_testing)



    print('Accuracy Score on Testing Data with C=', str(grid[i][0]), 'gamma =', str(grid[i][1]), ':', accuracy_score(Y_testing, Y_testing_pred))


# In[14]:


svc_rbf = SVC(kernel='rbf',gamma = 'scale',C=10)  # gamma='scale'
svc_rbf.fit(X_training,Y_training)


Y_training_pred = svc_rbf.predict(X_training)
Y_validation_pred = svc_rbf.predict(X_validation)
Y_testing_pred = svc_rbf.predict(X_testing)
print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))

svc_rbf = SVC(kernel='rbf',gamma = 'scale',C=100)  # gamma='scale'
svc_rbf.fit(X_training,Y_training)


Y_training_pred = svc_rbf.predict(X_training)
Y_validation_pred = svc_rbf.predict(X_validation)
Y_testing_pred = svc_rbf.predict(X_testing)

print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))

svc_rbf = SVC(kernel='rbf',gamma = 'auto',C=10)  # gamma='scale'
svc_rbf.fit(X_training,Y_training)


Y_training_pred = svc_rbf.predict(X_training)
Y_validation_pred = svc_rbf.predict(X_validation)
Y_testing_pred = svc_rbf.predict(X_testing)
print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))

svc_rbf = SVC(kernel='rbf',gamma = 'auto',C=100)  # gamma='scale'
svc_rbf.fit(X_training,Y_training)


Y_training_pred = svc_rbf.predict(X_training)
Y_validation_pred = svc_rbf.predict(X_validation)
Y_testing_pred = svc_rbf.predict(X_testing)

print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))


# In[15]:


gamma_vals = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
for i in range(len(gamma_vals)):
  svc_rbf = SVC(kernel='rbf',gamma = gamma_vals[i],C=10)  # gamma='scale'
  svc_rbf.fit(X_training,Y_training)


  #Y_training_pred = svc_rbf.predict(X_training)
  #Y_validation_pred = svc_rbf.predict(X_validation)
  Y_testing_pred = svc_rbf.predict(X_testing)
  print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))


# In[16]:


gamma_vals = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
for i in range(len(gamma_vals)):
    svc_rbf = SVC(kernel='rbf',gamma = gamma_vals[i],C=100)  # gamma='scale'
    svc_rbf.fit(X_training,Y_training)


    Y_testing_pred = svc_rbf.predict(X_testing)
    print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_rbf._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))


# ## SVM with polynomial kernel

# In[17]:


svc_poly = SVC(kernel='poly',degree=3,gamma=0.03,C=1) 
svc_poly.fit(X_training,Y_training)



Y_training_pred = svc_poly.predict(X_training)
Y_validation_pred = svc_poly.predict(X_validation)
Y_testing_pred = svc_poly.predict(X_testing)

print("SVM with Polynomial Kernel")

print('----------------------------------------------------------------------------------------------------')
print()
print('Tuned parameter Values, C :',svc_poly.C,', gamma :',svc_poly._gamma,', degree :',svc_poly.degree)
print()
print()
print('Accuracy Score on Training Data:',accuracy_score(Y_training, Y_training_pred))
print()
print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Validation Set--------------------------------------------')
print()
print('Accuracy Score on Validation Data:',accuracy_score(Y_validation, Y_validation_pred))
print()
#CM=confusion_matrix(Y_validation, Y_validation_pred)
print("Confusion Matrix on Validation Data:")
#print(CM)
plot_confusion_matrix(svc_poly, X_validation, Y_validation)  
plt.show()
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
print("Normalised Confusion Matrix on Validation Data:")
#print(CM_normalised)
plot_confusion_matrix(svc_poly, X_validation, Y_validation,normalize='true')  
plt.show()
print("Classification Report on Validation Data:")
print(classification_report(Y_validation,Y_validation_pred))
print()


print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Testing Set-----------------------------------------------')
print()
print('Accuracy Score on Testing Data:',accuracy_score(Y_testing, Y_testing_pred))
print()
print("Confusion Matrix on Test Data:")
#CM=confusion_matrix(Y_test, Y_test_pred)
#print(CM)
plot_confusion_matrix(svc_poly, X_test_mm, Y_test)  
plt.show()
print("Normalised Confusion Matrix on Test Data:")
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
#print(CM_normalised)
plot_confusion_matrix(svc_poly, X_test_mm, Y_test,normalize='true')  
plt.show()
print("Classification Report on Test Data:")
print(classification_report(Y_testing, Y_testing_pred))


# In[18]:


n = [2,3,4,5]
for i in range(len(n)):
  svc_poly = SVC(kernel='poly',degree=n[i]) 
  svc_poly.fit(X_training,Y_training)

  Y_testing_pred = svc_poly.predict(X_testing)

  print('Accuracy Score on Testing Data with degree =',n[i],':',accuracy_score(Y_testing, Y_testing_pred))


# In[19]:


trials = [0.001, 0.01, 0.1, 1, 10, 100]
#Make grid using above list
grid = []
for i in range(6):
    for j in range(4):
        grid.append((trials[i],trials[j]))

for i in range(len(grid)):
    svc_poly = SVC(kernel='poly',degree=3,gamma=grid[i][1],C=grid[i][0])  # gamma='scale'
    svc_poly.fit(X_training,Y_training)
    
    Y_testing_pred = svc_poly.predict(X_testing)



    print('Accuracy Score on Testing Data with C=', str(grid[i][0]), 'gamma =', str(grid[i][1]), ':', accuracy_score(Y_testing, Y_testing_pred))


# In[20]:


gamma_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
for i in range(len(gamma_vals)):
    svc_poly = SVC(kernel='poly',gamma = gamma_vals[i])  # gamma='scale'
    svc_poly.fit(X_training,Y_training)


    Y_testing_pred = svc_poly.predict(X_testing)
    print('Accuracy Score on Testing Data with C=', str(svc_rbf.C), 'gamma =', str(svc_poly._gamma), ':', accuracy_score(Y_testing, Y_testing_pred))


# ## SVM with sigmoid kernel

# In[21]:


svc_sigmoid = SVC(kernel='sigmoid',gamma=0.001,C=100) 
svc_sigmoid.fit(X_training,Y_training)



Y_training_pred = svc_sigmoid.predict(X_training)
Y_validation_pred = svc_sigmoid.predict(X_validation)
Y_testing_pred = svc_sigmoid.predict(X_testing)

print("SVM with Sigmoid Kernel")
print('----------------------------------------------------------------------------------------------------')
print()
print('Tuned parameter Values, C :',svc_sigmoid.C,', gamma :',svc_sigmoid._gamma)
print()
print()
print('Accuracy Score on Training Data:',accuracy_score(Y_training, Y_training_pred))
print()
print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Validation Set--------------------------------------------')
print()
print('Accuracy Score on Validation Data:',accuracy_score(Y_validation, Y_validation_pred))
print()
#CM=confusion_matrix(Y_validation, Y_validation_pred)
print("Confusion Matrix on Validation Data:")
#print(CM)
plot_confusion_matrix(svc_sigmoid, X_validation, Y_validation)  
plt.show()
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
print("Normalised Confusion Matrix on Validation Data:")
#print(CM_normalised)
plot_confusion_matrix(svc_sigmoid, X_validation, Y_validation,normalize='true')  
plt.show()
print("Classification Report on Validation Data:")
print(classification_report(Y_validation,Y_validation_pred))
print()


print('----------------------------------------------------------------------------------------------------')
print('--------------------------------------For Testing Set-----------------------------------------------')
print()
print('Accuracy Score on Testing Data:',accuracy_score(Y_testing, Y_testing_pred))
print()
print("Confusion Matrix on Test Data:")
#CM=confusion_matrix(Y_test, Y_test_pred)
#print(CM)
plot_confusion_matrix(svc_sigmoid, X_test_mm, Y_test)  
plt.show()
print("Normalised Confusion Matrix on Test Data:")
#CM_normalised = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
#print(CM_normalised)
plot_confusion_matrix(svc_sigmoid, X_test_mm, Y_test,normalize='true')  
plt.show()
print("Classification Report on Test Data:")
print(classification_report(Y_testing, Y_testing_pred))


# In[22]:


trials = [0.001, 0.01, 0.1, 1, 10, 100]
#Make grid using above list
grid = []
for i in range(6):
    for j in range(4):
        grid.append((trials[i],trials[j]))

for i in range(len(grid)):
    svc_rbf = SVC(kernel='sigmoid',gamma=grid[i][1],C=grid[i][0])  # gamma='scale'
    svc_rbf.fit(X_training,Y_training)
    
    Y_testing_pred = svc_rbf.predict(X_testing)



    print('Accuracy Score on Testing Data with C=', str(grid[i][0]), 'gamma =', str(grid[i][1]), ':', accuracy_score(Y_testing, Y_testing_pred))

