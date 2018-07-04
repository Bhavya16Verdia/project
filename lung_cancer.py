#!/usr/bin/python3

import pandas as pa
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#to read the data
data=pa.read_excel("data.xls").values
#print(data)
#print(data[0,1:24])

#training data
train_data=data[0:998,1:24]
train_target=data[0:998,24]
#print(train_target)

#testing data
test_data=data[999:,1:24]
test_target=data[999:,24]
#print('actual: ',test_target)

#calling the decesion tree algo
clf=DecisionTreeClassifier()
trained=clf.fit(train_data,train_target)

#calling SVM algo
clf1=SVC()
trained1=clf1.fit(train_data,train_target)

#calling KNN algo
clf2=KNeighborsClassifier(n_neighbors=3)
trained2=clf2.fit(train_data,train_target)

#empty list to store the user inputs
a=[]

#list to store the questions
b=["Enter your age","Enter your gender (*1 for male *2 for female)","Enter the level of air pollution in your surroundings","enter the amount of your alcohol consumption","enter the level of dust allergy you suffer from","enter the level of occupational hazards you face","rate the amount of genetic risk","enter the level of chronic lung disease you face(if any)","rate your diet","enter the levels of obesity","enter the levels of smoking","enter the level of passive smoking","how often do you have the problem of chest pain","how often do you have the problem of coughing blood","rate the level of fatigueness","rate the weight loss in your body","how often do you experience shortness of breath","how often have u noticed wheezing","enter the level of difficulty you feel during swallowing food","how often do you notice clubbing of finger nails","how frequently do you have cold","how frequently do you have dry cough","rate the level of snoring"]

#loop for the questions to be answered
for i in range(len(b)):
	print (b[i])
	x=int(input())
	a.append(x)
print(a)

#displaying the result
predicted=trained.predict(a)
#predicted1=trained1.predict(test)
#predicted2=trained2.predict(test)

print(predicted)
#print(predicted1)
#print(predicted2)

#print(test_target)
'''
acc=accuracy_score(predicted,test_target)
print(acc)
acc1=accuracy_score(predicted1,test_target)
print(acc1)
acc2=accuracy_score(predicted2,test_target)
print(acc2)
#print(train_target)
'''
