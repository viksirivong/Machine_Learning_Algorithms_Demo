#-------------------------------------------------------------------------
# AUTHOR: Thongsavik Sirivong
# FILENAME: decision_tree.py
# SPECIFICATION: The program train, test, and output the performance of the models created by using each training set from files, contact_lens_training_1.csv, contact_lens_training_2.csv, 
#                and contact_lens_training_3.csv, on the test set provided (contact_lens_test.csv).
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from os import access
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

   dbTraining = []
   X = []
   Y = []

   #reading the training data in a csv file
   with open(ds, 'r') as csvfile:
      reader = csv.reader(csvfile)
      for i, row in enumerate(reader):
         if i > 0: #skipping the header
            dbTraining.append (row)

   #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
   #--> add your Python code here
   for i, row in enumerate(dbTraining):
      X.append(row.copy()[0:4])
      for j, instance in enumerate(row):
         if j == 0:
            if instance == 'Young':
               X[i][j] = 1
            elif instance == 'Prepresbyopic':
               X[i][j] = 2
            elif instance == 'Presbyopic':
               X[i][j] = 3
         elif j == 1:
            if instance == 'Myope':
               X[i][j] = 1
            elif instance == 'Hypermetrope':
               X[i][j] = 2
         elif j == 2:
            if instance == 'Yes':
               X[i][j] = 1
            elif instance == 'No':
               X[i][j] = 2
         elif j == 3:
            if instance == 'Normal':
               X[i][j] = 1
            elif instance == 'Reduced':
               X[i][j] = 2

   #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
   #--> add your Python code here
   for i, row in enumerate(dbTraining):
      if row[4] == 'Yes':
         Y.append(1)
      else:
         Y.append(2)

   lowestAccuracy = 1

   #loop your training and test tasks 10 times here
   for i in range (10):

      #fitting the decision tree to the data setting max_depth=3
      clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
      clf = clf.fit(X, Y)

      #read the test data and add this data to dbTest
      #--> add your Python code here
      dbTest = []
      with open('contact_lens_test.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
            if i > 0: #skipping the header
               dbTest.append (row)

      TP_TN = 0
      for data in dbTest:
         #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
         #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
         #--> add your Python code here
         if data[0] == 'Young':
            data[0] = 1
         elif data[0] == 'Prepresbyopic':
            data[0] = 2
         elif data[0] == 'Presbyopic':
            data[0] = 3
         if data[1] == 'Myope':
            data[1] = 1
         elif data[1] == 'Hypermetrope':
            data[1] = 2
         if data[2] == 'Yes':
            data[2] = 1
         elif data[2] == 'No':
            data[2] = 2
         if data[3] == 'Normal':
            data[3] = 1
         elif data[3] == 'Reduced':
            data[3] = 2
         if data[4] == 'Yes':
            data[4] = 1
         elif data[4] == 'No':
            data[4] = 2   

         class_predicted = clf.predict([data[0:4]])[0]

         #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
         #--> add your Python code here
         if class_predicted == data[4]:
            TP_TN += 1

      #find the lowest accuracy of this model during the 10 runs (training and test set)
      #--> add your Python code here
      accuracy = TP_TN / len(dbTest)
      if accuracy < lowestAccuracy:
         lowestAccuracy = accuracy

   #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
   #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
   #--> add your Python code here
   print('Final accuracy when training on', ds, ':', lowestAccuracy)
   