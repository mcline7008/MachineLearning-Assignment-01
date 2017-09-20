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
import matplotlib.pyplot as plt

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

def costFunctionLinear(theta, trainingData):
    ''' Compute the cost function for the linear regression '''

    # number of values used in the training
    m = len(trainingData)

    #list to store the predictions
    predictions = []

    # running sum of the squared error values
    errorSum = 0

    # create predictions using the current theta and calculate error
    for student in trainingData:
        prediction = theta[0] + theta[1] * student.knowlTrans
        squaredError = (prediction - student.risk)**2
        predictions.append(prediction)
        errorSum += squaredError

    # calculate the mean squared error
    return errorSum  / (2*m)

def costFunctionQuadratic(theta, trainingData):
    ''' Compute the cost function for the linear regression '''

    # number of values used in the training
    m = len(trainingData)

    #list to store the predictions
    predictions = []

    # running sum of the squared error values
    errorSum = 0

    # create predictions using the current theta and calculate error
    for student in trainingData:
        prediction = theta[0] + theta[1] * student.knowlTrans + theta[2] * (student.knowlTrans ** 2)
        squaredError = (prediction - student.risk)**2
        predictions.append(prediction)
        errorSum += squaredError

    # calculate the mean squared error
    return errorSum  / (2*m)

def costFunctionMultiVar(theta, trainingData):
    ''' Compute the cost function for the linear regression with 2 features'''

    # number of values used in the training
    m = len(trainingData)

    #list to store the predictions
    predictions = []

    #running sum of the squared error values
    errorSum = 0

    #create predictions using the current theta and calculate the error
    for student in trainingData:
        prediction = theta[0] + theta[1] * student.knowlTrans + theta[2] * student.respEtiq
        squaredError = (prediction - student.risk) ** 2
        predictions.append(prediction)
        errorSum += squaredError

    # calculate the mean squared error
    return errorSum / (2*m)


def gradientDescentLinear(theta0, alpha, trainingData, maxIterations):
    ''' Perform gradient descent to minimize the cost function value to an acceptable point or stops at maxIterations '''
    costHistory = []
    thetaHistory = []
    theta = theta0
    iterations = []
    previousCost = 0

    for i in range(maxIterations):
        predictions = []
        errorSum0 = 0
        errorSum1 = 0

        for student in trainingData:
            prediction = theta[0] + theta[1] * student.knowlTrans
            error0 = prediction - student.risk
            error1 = (prediction - student.risk) * student.knowlTrans
            predictions.append(prediction)
            errorSum0 += error0
            errorSum1 += error1

        thetaHistory.append(theta)
        theta[0] = theta[0] - alpha * (1.0 / len(trainingData)) * errorSum0
        theta[1] = theta[1] - alpha * (1.0 / len(trainingData)) * errorSum1

        currentCost = costFunctionLinear(theta, trainingData)

        iterations.append(i)

        if abs(currentCost - previousCost) < 0.00000001:
            costHistory.append(currentCost)
            break

        previousCost = currentCost
        costHistory.append(currentCost)

    plt.plot(iterations, costHistory, 'o')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()

    return theta, len(iterations)

def gradientDescentQuadratic(theta0, alpha, trainingData, maxIterations):
    ''' Perform gradient descent to minimize the cost function value to an acceptable point or stops at maxIterations '''
    costHistory = []
    thetaHistory = []
    theta = theta0
    iterations = []
    previousCost = 0

    for i in range(maxIterations):
        predictions = []
        errorSum0 = 0
        errorSum1 = 0
        errorSum2 = 0

        for student in trainingData:
            prediction = theta[0] + theta[1] * student.knowlTrans + theta[2] * (student.knowlTrans ** 2)
            error0 = prediction - student.risk
            error1 = (prediction - student.risk) * student.knowlTrans
            error2 = (prediction - student.risk) * (student.knowlTrans ** 2)
            predictions.append(prediction)
            errorSum0 += error0
            errorSum1 += error1
            errorSum2 += error2

        thetaHistory.append(theta)
        theta[0] = theta[0] - alpha * (1.0 / len(trainingData)) * errorSum0
        theta[1] = theta[1] - alpha * (1.0 / len(trainingData)) * errorSum1
        theta[2] = theta[2] - alpha * (1.0 / len(trainingData)) * errorSum2

        currentCost = costFunctionQuadratic(theta, trainingData)

        iterations.append(i)

        if abs(currentCost - previousCost) < 0.00000001:
            costHistory.append(currentCost)
            break

        previousCost = currentCost
        costHistory.append(currentCost)

    plt.plot(iterations, costHistory, 'o')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()

    return theta, len(iterations)


def gradientDescentMultiVar(theta0, alpha, trainingData, maxIterations):
    ''' Perform gradient descent to minimize the cost function associated with a multi-variable regression model. '''

    costHistory = []
    thetaHistory = []
    theta = theta0
    iterations = []
    previousCost = 0

    for i in range(maxIterations):
        predictions = []
        errorSum0 = 0
        errorSum1 = 0
        errorSum2 = 0

        for student in trainingData:
            prediction = theta[0] + theta[1] * student.knowlTrans + theta[2] * student.respEtiq
            error0 = prediction - student.risk
            error1 = (prediction - student.risk) * student.knowlTrans
            error2 = (prediction - student.risk) * student.respEtiq
            predictions.append(prediction)
            errorSum0 += error0
            errorSum1 += error1
            errorSum2 += error2

        thetaHistory.append(theta)
        theta[0] = theta[0] - alpha * (1.0 / len(trainingData)) * errorSum0
        theta[1] = theta[1] - alpha * (1.0 / len(trainingData)) * errorSum1
        theta[2] = theta[2] - alpha * (1.0 / len(trainingData)) * errorSum2

        currentCost = costFunctionLinear(theta, trainingData)

        iterations.append(i)

        if abs(currentCost - previousCost) < 0.00000001:
            costHistory.append(currentCost)
            break

        previousCost = currentCost
        costHistory.append(currentCost)

    plt.plot(iterations, costHistory, 'o')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()

    return theta, len(iterations)


####### GLOBAL VARIABLES #######
alpha = 0.4
thetaInit = [2,2]
trainingPercentage = 0.2


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

####### PARTITION THE DATA INTO TRAINING AND TEST #######
trainingData = []
testData = []
trainingSamples = len(cleanData) * trainingPercentage

counter = 0
for student in cleanData:
    if counter < trainingSamples:
        trainingData.append(student)
    else:
        testData.append(student)
    counter+=1

print("####### PREPARING DATA #######")
print('Training Samples: ', len(trainingData))
print('Test Samples: ', len(testData))
print("\n\n\n")


####### DUMMY DATASET TO EXPERIMENT WITH #######
dummyStudents = []
for i in range(5):
    dummyStudents.append(Student(i,i,i+1))

'''
thetaFinal = gradientDescentLinear(thetaInit, alpha, dummyStudents, 10000)
print("Theta Final: ", thetaFinal)
'''

####### LINEAR REGRESSION WITH ONE VARIABLE #######
print("######## TESTING LINEAR MODEL #######")
thetaFinal, iterations = gradientDescentLinear(thetaInit, alpha, trainingData, 100000)
print('Iterations Needed for Convergence: ', iterations)
print('Theta Final:', thetaFinal)
predictionError = costFunctionLinear(thetaFinal, testData)
print('Prediciton Error:', predictionError)
print("\n\n\n")

####### QUADRATIC REGRESSION WITH ONE VARIABLE #######
print("####### TESTING QUADRATIC MODEL #######")
theta = [0,0,0]
thetaFinal, iterations = gradientDescentQuadratic(theta, alpha, trainingData, 100000)
print('Iterations Needed for Convergence: ', iterations)
print('Theta Final: ', thetaFinal)
predictionError = costFunctionQuadratic(thetaFinal, testData)
print('Prediction Error: ', predictionError)
print("\n\n\n")

####### MULTI-VARIABLE LINEAR REGRESSION #######
print("####### TESTING MULTI_VARIABLE LINEAR REGRESSION #######")
theta = [0,0,0]
thetaFinal, iterations = gradientDescentMultiVar(theta, alpha, dummyStudents, 100000)
print('Iterations Needed for Convergence: ', iterations)
print('Theta Final: ', thetaFinal)
predictionError = costFunctionMultiVar(thetaFinal, testData)
print('Prediction Error: ', predictionError)
print("\n\n\n")


######## VISUALIZATIONS ########

