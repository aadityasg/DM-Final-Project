# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:50:26 2017

@author: Odin1
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.metrics import mean_squared_error
    
def readFile(numberOfLatentFactors, userRegularizationFactor, itemRegularizationFactor, userBiasFactor, itemBiasFactor):
    
    print ('Started reading file')
    df = pd.read_csv('ratings_processed.csv')
    #df = pd.read_csv('subset.csv')
    print ('File read complete')
    
    user_list = df['user_id'].drop_duplicates().tolist()
    user_list.sort()
    movie_list = df['movie_id'].drop_duplicates().tolist()
    movie_list.sort()
    print ('Item sorting completed')

    numberOfUsers = len(user_list)
    numberOfItems = len(movie_list)
    ratingMatrix = np.zeros((numberOfUsers, numberOfItems))
    
    print ('Number of users : {}'.format(numberOfUsers))
    print ('Number of movies : {}'.format(numberOfItems))
    print ('Size of df : {}'.format(len(df)))
    
    for row in range(len(df)):
        ratingMatrix[int(df.iloc[row]['user_id']), int(df.iloc[row]['movie_id'])] = df.iloc[row]['rating']
    
    iterArray = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    calculateLearningCurve(iterArray, 0.1, numberOfLatentFactors, userRegularizationFactor, itemRegularizationFactor, 
                           userBiasFactor, itemBiasFactor, ratingMatrix, numberOfUsers, numberOfItems)
    
    
def trainingStep(numberOfIterations, mock_len, mock_row, mock_col, ratingMatrix, userBias, learningRate, userBiasFactor, itemBias, itemBiasFactor,
                 userVector, itemVector, userRegularizationFactor,itemRegularizationFactor, globalBias ):
    for i in range(1, numberOfIterations + 1):
        randomIndices = np.arange(mock_len)
        np.random.shuffle(randomIndices)
        for j in randomIndices:
            user = mock_row[j]
            item = mock_col[j]
            
            val = globalBias + userBias[user] + itemBias[item]
            val += userVector[user, :].dot(itemVector[item, :].T)
            
            prediction = val
            error = ratingMatrix[user, j] - prediction
                                     
            userBias[user] += learningRate * (error - userBiasFactor * userBias[user])
            itemBias[item] += learningRate * (error - itemBiasFactor * itemBias[item])
                             
            userVector[user, :] += learningRate * (error * itemVector[item, :] - userRegularizationFactor * userVector[user, :])
            itemVector[item, :] += learningRate * (error * userVector[user, :] - itemRegularizationFactor * itemVector[item, :])
         
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual) 

def findAll(userVector, itemVector, globalBias, userBias, itemBias):
    values = np.zeros((userVector.shape[0], itemVector.shape[0]))
    for u in xrange(userVector.shape[0]):
        for i in xrange(itemVector.shape[0]):
            val = globalBias + userBias[u] + itemBias[i]
            val += userVector[u, :].dot(itemVector[i, :].T)
            values[u, i] = val
    return values

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=10, replace=False)

    train[user, test_ratings] = 0.
    test[user, test_ratings] = ratings[user, test_ratings]

    return train, test

def calculateLearningCurve(iterArray, learningRate, numberOfLatentFactors, userRegularizationFactor, 
                           itemRegularizationFactor, userBiasFactor, itemBiasFactor, ratingMatrix, numberOfUsers, numberOfItems):
    iterArray.sort()
    trainMSE = []
    testMSE = []
    iter = 0
    
    mock_col, mock_row = ratingMatrix.nonzero()
    mock_len = len(mock_col)

    train, test = train_test_split(ratingMatrix)
    
    userVector = np.random.normal(scale=1./numberOfLatentFactors, size=(numberOfUsers, numberOfLatentFactors))
    itemVector = np.random.normal(scale=1./numberOfLatentFactors, size=(numberOfItems, numberOfLatentFactors))
    
    userBias = np.zeros(numberOfUsers)
    itemBias = np.zeros(numberOfItems)
    
    globalBias = np.mean(np.nonzero(ratingMatrix))
    
    for(i, iterN) in enumerate(iterArray):
        print ('Iteration: {}'.format(iterN))
        trainingStep(iterN - iter, mock_len, mock_row, mock_col, ratingMatrix, userBias, learningRate, userBiasFactor, itemBias, itemBiasFactor, 
                     userVector, itemVector, userRegularizationFactor,itemRegularizationFactor, globalBias )

    findings = findAll(userVector, itemVector, globalBias, userBias, itemBias)
    
    pred_max, pred_min = findings.max(), findings.min()
    findings = (findings - pred_min) / (pred_max - pred_min)

    for u in xrange(numberOfUsers):
        for i in xrange(numberOfItems):
            findings[u, i] = round(findings[u, i] * 5, 0)
    
    """for index1 in xrange(self.numberOfUsers):
        for index2 in xrange(self.numberOfItems):
            if findings[index1][index2] > 4.1:
                print('{:4}'.format(self.ratingMatrix[index1][index2]))
                print('{:4}'.format(findings[index1][index2]))
                print(' ')"""
    
    trainMSE += [get_mse(findings, train)]
    testMSE += [get_mse(findings, test)]
    
    print ('Train mse: ' + str(trainMSE[-1]))
    print ('Test mse: ' + str(testMSE[-1]))
    
    iter = iterN
    
    hist = np.histogram(findings.flatten() , bins=[0,1,2,3,4,5,6])
    print(hist)
    
readFile(40, 0.3, 0.3, 0.3, 0.3)
