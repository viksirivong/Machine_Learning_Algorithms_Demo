from sklearn import tree
import csv

dbTest = []

with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)

for data in dbTest:
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

    print(data)