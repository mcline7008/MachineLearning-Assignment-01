###################################################################
##
## Machine Learning Assignment 1
## Regression Fundamentals
##
## @author: Matthew Cline
##
###################################################################

import pandas as pd
import math

class Student(object):
    """A student from the flu survey data set with the following properties:

    Attributes:
        knowlTrans: the knowlege of transmission value
        risk: the risk value
        respEtiq: the respitory etiquette value
    """
    def __init__(self, knowlTrans, risk, respEtiq):
        """Return a Student object with the specified attributes"""
        self.knowlTrans = knowlTrans
        self.risk = risk
        self.respEtiq = respEtiq

def costFunction(theta, x, y):
    #enter code here
    pass

def gradientDecent(theta0, alpha, x, y):
    #enter code here
    pass

def mean(values):
    return sum(values) / float(len(values))


####### IMPORT DATA FROM EXCEL INTO PANDAS DATAFRAME #######
data = pd.read_excel('fluML.xlsx', sheetname='Sheet1')
cleanData = []


####### CLEAN THE DATA AND ORGANIZE #######
i = 0
for student in data['Student']:
    knowlTrans = data['KnowlTrans'][i]
    risk = data['Risk'][i]
    respEtiq = data["RespEtiq"][i]
    i += 1
    ''' Check to ensure that all of the necessary values in the record are populated '''
    if(math.isnan(knowlTrans) or math.isnan(risk) or math.isnan(respEtiq)):
        continue
    ''' Check to ensure that the RespEtiq values provided are valid. If the values are invalid, the record will be thrown out. '''
    if(respEtiq < 1 or respEtiq > 5):
        continue
    cleanData.append(Student(knowlTrans, risk, respEtiq))


print(len(cleanData))


####### DUMMY DATASET TO EXPERIMENT WITH #######
dummyData = [[1,1], [2,3], [3,2], [4,3], [5,5]]
dummyX = [row[0] for row in dummyData]
dummyY = [row[1] for row in dummyData]

#print(mean(dummyX))
#print(mean(dummyY))


#print("Average Vaccine: ", mean(data['Vaccin']))
