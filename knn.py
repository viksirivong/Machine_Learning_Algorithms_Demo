#-------------------------------------------------------------------------
# AUTHOR: Thongsavik Sirivong
# FILENAME: knn.py
# SPECIFICATION: The program reads the file binary_points.csv and output the LOO-CV error rate for 1NN.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('data/binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#the number of wrong prediction to calculate the error rate
wrongPrediction = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    for j, trainingFeatures in enumerate(db):
        if j != i:
            feature = trainingFeatures.copy()[0:2]
            feature[0] = float(feature[0])
            feature[1] = float(feature[1])
            X.append(feature)
                
    #print(X)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for j, trainingClass in enumerate(db):
        if j != i:
            if trainingClass[2] == '-':
                Y.append(1.0)
            elif trainingClass[2] == '+':
                Y.append(2.0)
    
    #print(Y)

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = instance.copy()
    testSample[0] = float(testSample[0])
    testSample[1] = float(testSample[1])
    if testSample[2] == '-':
        testSample[2] = 1.0
    else:
        testSample[2] = 2.0
    
    #print(testSample)

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample[0:2]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != testSample[2]:
        wrongPrediction += 1

#print the error rate
#--> add your Python code here
errorRate = wrongPrediction / len(db)
print ('Error Rate = ', errorRate, ' or ', errorRate * 100, '%')
