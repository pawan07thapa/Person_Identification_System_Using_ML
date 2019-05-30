import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np



img=cv2.imread("./att_faces/s1/1.pgm",0)
print(img.shape)


fig,ax=plt.subplots(1,3,figsize=(10,5))
equ = cv2.equalizeHist(img)
blur=cv2.blur(equ,(3,3))
ax[0].imshow(img,cmap='gray')
ax[1].imshow(equ,cmap='gray')
ax[2].imshow(blur,cmap='gray')
plt.show()



for xml_file in ['haarcascade_frontalface_default.xml','haarcascade_frontalface_alt.xml','lbpcascade_frontalface_improved.xml']:
    face_cascade = cv2.CascadeClassifier(xml_file)
    tp,fp,fn=[],[],[]
    for people in range(1,41):
        if people==34:
            continue
        for imgno in range(1,11):
            img=cv2.imread('./att_faces/s{}/{}.pgm'.format(people,imgno),0)
            assert img.shape==(112,92)
            equ = cv2.equalizeHist(img)
            blur=cv2.blur(equ,(3,3))
            faces = face_cascade.detectMultiScale(blur)
            if len(faces)==0:
                fn.append((people,imgno))
            elif len(faces)==1:
                x,y,w,h=faces[0]
                if w*h>2300:
                    tp.append((people,imgno))
                else:
                    fp.append((people,imgno))
            else:
                print("Multiple faces detected")

    print("Performance metrics for {}:".format(xml_file))
    print("1. Total:{}".format(390))
    print("2. True positive:{}".format(len(tp)))
    print("3. False positive:{}".format(len(fp)))
    print("4. False negative:{}".format(len(fn)))
    print("5. Accuracy:{}".format(len(tp)/390))



block_size,count=50,0
dataset=np.zeros((375,block_size**2+1),dtype=np.uint8)
for people in range(1,41):
    if people==34:
        continue
    for imgno in range(1,11):
        img=cv2.imread('./att_faces/s{}/{}.pgm'.format(people,imgno),0)
        assert img.shape==(112,92)
        equ = cv2.equalizeHist(img)
        blur=cv2.blur(equ,(3,3))
        faces = face_cascade.detectMultiScale(blur)
        if len(faces)==1:
            x,y,w,h=faces[0]
            reduced_img=img[x:x+w,y:y+h]
            reduced_img=cv2.resize(blur,(block_size,block_size))
            reduced_img.shape=(1,block_size**2)
            assert reduced_img.shape==(1,block_size**2)
            dataset[count,:block_size**2]=reduced_img
            dataset[count,block_size**2]=people
            count+=1
            
df=pd.DataFrame(dataset)
df.to_csv('dataset.csv')


os.mkdir('./processed_dataset')
for people in range(1,41):
    os.mkdir('./processed_dataset/s{}'.format(people))
    for imgno in range(1,11):
        img=cv2.imread('./att_faces/s{}/{}.pgm'.format(people,imgno),0)
        assert img.shape==(112,92)
        equ = cv2.equalizeHist(img)
        blur=cv2.blur(equ,(3,3))
        cv2.imwrite('./processed_dataset/s{}/{}.jpg'.format(people,imgno),blur)



from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
#import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn.ensemble import RandomForestClassifier # A combine model of many decision t

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# np.set_printoptions(threshold=np.nan)





# import pandas as pd
# df=pd.read_csv("dataset.csv")
df.head()


df[2500].value_counts()


X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,0:block_size**2],df.iloc[:,block_size**2] , test_size = 0.25)

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
print(Y_train.value_counts())


import random

train_set_frac=0.8

training_set_index,test_set_index=set(),set()

freq_count=dict(df[2500].value_counts())
for identity,freq in freq_count.items():
    training_samples=int(freq*train_set_frac)
#     print(identity,freq,training_samples)
    training_set_index.update(set(random.sample(set(df.index[df[2500]==identity]),training_samples)))
    
test_set_index=set(df.index).difference(training_set_index)
X_train=df.iloc[list(training_set_index),:block_size**2]
Y_train=df.iloc[list(training_set_index),block_size**2]
X_test=df.iloc[list(test_set_index),:block_size**2]
Y_test=df.iloc[list(test_set_index),block_size**2]

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)


#Random Forest
clf1 = RandomForestClassifier(n_estimators=100 ,random_state=15)
clf1.fit(X_train, Y_train)
pre = clf1.predict(X_test)

#Saving model
filename = 'random_forest_model.sav'
joblib.dump(clf1, filename)

#Using Current Classfier
pre=clf1.predict(X_test)

#Printing the accuracy score
print("Accuracy of Random Forest is:{}".format(accuracy_score(Y_test,pre)))
# print("Confusion matrix:\n")
# cm=pd.DataFrame(confusion_matrix(Y_test, pre))
# cm


import random
lst=random.sample(list(X_test.index),8)
fig, ax = plt.subplots(2,4,figsize=(10,7))
fig.suptitle("Results for Random Forest algo:")
for ind in range(8):
    pics=X_test.loc[lst[ind],:].values
    pics.shape=(50,50)
    ax[ind//4,ind%4].set_title("{}:{}".format(Y_test[lst[ind]],clf1.predict(X_test.loc[lst[ind],:])[0]))
    ax[ind//4,ind%4].imshow(pics,cmap='gray')
plt.show()


#decison tree classifier
clf2 = DecisionTreeClassifier()
clf2.fit(X_train,Y_train)
pre=clf2.predict(X_test)

#Saving model
filename = 'Decison_Tree_model.sav'
joblib.dump(clf2, filename)

#Using Current Classfier
pre=clf2.predict(X_test)

#Printing the accuracy score
print("Accuracy of Decision Tree is:{}".format(accuracy_score(Y_test,pre)))
# print("Confusion matrix:\n")
# cm=pd.DataFrame(confusion_matrix(Y_test, pre))
# cm


import random
lst=random.sample(list(X_test.index),8)
fig, ax = plt.subplots(2,4,figsize=(10,7))
fig.suptitle("Results for Decision Tree algo:")
for ind in range(8):
    pics=X_test.loc[lst[ind],:].values
    pics.shape=(50,50)
    ax[ind//4,ind%4].set_title("{}:{}".format(Y_test[lst[ind]],clf2.predict(X_test.loc[lst[ind],:])[0]))
    ax[ind//4,ind%4].imshow(pics,cmap='gray')
plt.show()


#SVM classifier
clf3 = svm.SVC(kernel='linear',C=1)
clf3.fit(X_train,Y_train)
pre=clf3.predict(X_test)

#Saving model
filename = 'SVM_model.sav'
joblib.dump(clf3, filename)

#Using Current Classfier
pre=clf3.predict(X_test)

#Printing the accuracy score
print("Accuracy of SVM is:{}".format(accuracy_score(Y_test,pre)))
# print("Confusion matrix:\n")
# cm=pd.DataFrame(confusion_matrix(Y_test, pre))
# cm


import random
lst=random.sample(list(X_test.index),8)
fig, ax = plt.subplots(2,4,figsize=(10,7))
fig.suptitle("Results for SVM algo:")
for ind in range(8):
    pics=X_test.loc[lst[ind],:].values
    pics.shape=(50,50)
    ax[ind//4,ind%4].set_title("{}:{}".format(Y_test[lst[ind]],clf3.predict(X_test.loc[lst[ind],:])[0]))
    ax[ind//4,ind%4].imshow(pics,cmap='gray')
plt.show()


#Logistic regression classifier
clf4 = LogisticRegression()
clf4.fit(X_train,Y_train)
pre=clf4.predict(X_test)

#Saving model
filename = 'Logistic_regression_model.sav'
joblib.dump(clf4, filename)

#Using Current Classfier
pre=clf4.predict(X_test)

#Printing the accuracy score
print("Accuracy of Logistic Regression is:{}".format(accuracy_score(Y_test,pre)))
# print("Confusion matrix:\n")
# cm=pd.DataFrame(confusion_matrix(Y_test, pre))
# cm


import random
lst=random.sample(list(X_test.index),8)
fig, ax = plt.subplots(2,4,figsize=(10,7))
fig.suptitle("Results for Logistics Regression algo:")
for ind in range(8):
    pics=X_test.loc[lst[ind],:].values
    pics.shape=(50,50)
    ax[ind//4,ind%4].set_title("{}:{}".format(Y_test[lst[ind]],clf4.predict(X_test.loc[lst[ind],:])[0]))
    ax[ind//4,ind%4].imshow(pics,cmap='gray')
plt.show()


#Logistic regression classifier
clf5 = GaussianNB()
clf5.fit(X_train,Y_train)
pre=clf5.predict(X_test)

#Saving model
filename = 'Logistic_regression_model.sav'
joblib.dump(clf5, filename)

#Using Current Classfier
pre=clf5.predict(X_test)

#Printing the accuracy score
print("Accuracy of Naive Bayes is:{}".format(accuracy_score(Y_test,pre)))
# print("Confusion matrix:\n")
# cm=pd.DataFrame(confusion_matrix(Y_test, pre))
# cm



import random
lst=random.sample(list(X_test.index),8)
fig, ax = plt.subplots(2,4,figsize=(10,7))
fig.suptitle("Results for Naive Bayes algo:")
for ind in range(8):
    pics=X_test.loc[lst[ind],:].values
    pics.shape=(50,50)
    ax[ind//4,ind%4].set_title("{}:{}".format(Y_test[lst[ind]],clf5.predict(X_test.loc[lst[ind],:])[0]))
    ax[ind//4,ind%4].imshow(pics,cmap='gray')
plt.show()

