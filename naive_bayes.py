#-------------------------------------------------------------------------
# AUTHOR: Thongsavik Sirivong
# FILENAME: naive_bayes.py
# SPECIFICATION: The program reads the file weather_training.csv (training set) and output the classification of each test instance from the file weather_test (test set) if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data
#--> add your Python code here
dbTraining = []

#reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for i, row in enumerate(dbTraining):
    X.append(row.copy()[1:5])
    for j, instance in enumerate(row[1:5]):
        if j == 0:
            if instance == 'Sunny':
                X[i][j] = 1
            elif instance == 'Overcast':
                X[i][j] = 2
            elif instance == 'Rain':
                X[i][j] = 3
        elif j == 1:
            if instance == 'Hot':
                X[i][j] = 1
            elif instance == 'Mild':
                X[i][j] = 2
            elif instance == 'Cool':
                X[i][j] = 3
        elif j == 2:
            if instance == 'High':
                X[i][j] = 1
            elif instance == 'Normal':
                X[i][j] = 2
        elif j == 3:
            if instance == 'Weak':
                X[i][j] = 1
            elif instance == 'Strong':
                X[i][j] = 2

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for i, row in enumerate(dbTraining):
    if row[5] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
dbTest = []

#reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)

#transform the original test features to numbers and add to vector testSample
testSample = []
for i, row in enumerate(dbTest):
    testSample.append(row.copy()[1:6])
    for j, instance in enumerate(row[1:6]):
        if j == 0:
            if instance == 'Sunny':
                testSample[i][j] = 1
            elif instance == 'Overcast':
                testSample[i][j] = 2
            elif instance == 'Rain':
                testSample[i][j] = 3
        elif j == 1:
            if instance == 'Hot':
                testSample[i][j] = 1
            elif instance == 'Mild':
                testSample[i][j] = 2
            elif instance == 'Cool':
                testSample[i][j] = 3
        elif j == 2:
            if instance == 'High':
                testSample[i][j] = 1
            elif instance == 'Normal':
                testSample[i][j] = 2
        elif j == 3:
            if instance == 'Weak':
                testSample[i][j] = 1
            elif instance == 'Strong':
                testSample[i][j] = 2

#printing the header of the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
for i, instance in enumerate(testSample):
    predicted = clf.predict_proba([instance[0:4]])[0]
    if predicted[0] > predicted[1]:
        dbTest[i][5] = 'Yes'
        instance[4] = predicted[0]
    else:
        dbTest[i][5] = 'No'
        instance[4] = predicted[1]

#printing the solution
for i, instance in enumerate(dbTest):
    if testSample[i][4] >= 0.75:
        print(str(instance[0]).ljust(15) + str(instance[1]).ljust(15) + str(instance[2]).ljust(15) + str(instance[3]).ljust(15) + str(instance[4]).ljust(15) + str(instance[5]).ljust(15) + str(testSample[i][4]).ljust(15))
