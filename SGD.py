# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:50:26 2017

@author: Odin1
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.metrics import mean_squared_error

class SGDC():

    def __init__(self, numberOfLatentFactors, userRegularizationFactor, itemRegularizationFactor, userBiasFactor, itemBiasFactor):
        #self.numberOfUsers, self.numberOfItems = ratingMatrix.shape
        self.numberOfLatentFactors = numberOfLatentFactors
        self.userRegularizationFactor = userRegularizationFactor
        self.itemRegularizationFactor = itemRegularizationFactor
        self.userBiasFactor = userBiasFactor
        self.itemBiasFactor = itemBiasFactor
        self.readFile()
        
    def readFile(self):
        
        print ('Started reading file')
        df = pd.read_csv('ratings_processed.csv')
        #df = pd.read_csv('subset.csv')
        print ('File read complete')
        
        user_list = df['user_id'].drop_duplicates().tolist()
        user_list.sort()
        movie_list = df['movie_id'].drop_duplicates().tolist()
        movie_list.sort()
        print ('Item sorting completed')

        self.numberOfUsers = len(user_list)
        self.numberOfItems = len(movie_list)
        self.ratingMatrix = np.zeros((self.numberOfUsers, self.numberOfItems))
        
        print ('Number of users : {}'.format(self.numberOfUsers))
        print ('Number of movies : {}'.format(self.numberOfItems))
        print ('Size of df : {}'.format(len(df)))
        
        self.mock_col, self.mock_row = self.ratingMatrix.nonzero()
        self.mock_len = len(self.mock_col)

        for row in range(len(df)):
            self.ratingMatrix[int(df.iloc[row]['user_id']), int(df.iloc[row]['movie_id'])] = df.iloc[row]['rating']
        
        iterArray = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.calculateLearningCurve(iterArray, 0.1)
    
    def training(self, numberOfIterations = 10, learningRate = 0.1):
        """ Randomly create user and feature vectors """
        self.userVector = np.random.normal(scale=1./self.numberOfLatentFactors, size=(self.numberOfUsers, self.numberOfLatentFactors))
        self.itemVector = np.random.normal(scale=1./self.numberOfLatentFactors, size=(self.numberOfItems, self.numberOfLatentFactors))
        self.learningRate = learningRate
        
        self.userBias = np.zeros(self.numberOfUsers)
        self.itemBias = np.zeros(self.numberOfItems)
        
        #print ', '.join(map(str, self.itemBias))
        
        self.globalBias = np.mean(np.nonzero(self.ratingMatrix))
        self.trainingStep(numberOfIterations)
        
    def trainingStep(self, numberOfIterations):
        for i in range(1, numberOfIterations + 1):
            randomIndices = np.arange(self.mock_len)
            np.random.shuffle(randomIndices)
            for j in randomIndices:
                user = self.mock_row[j]
                item = self.mock_col[j]
                prediction = self.find(user, item)
                error = self.ratingMatrix[user, j] - prediction
                                         
                self.userBias[user] += self.learningRate * (error - self.userBiasFactor * self.userBias[user])
                self.itemBias[item] += self.learningRate * (error - self.itemBiasFactor * self.itemBias[item])
                                 
                self.userVector[user, :] += self.learningRate * (error * self.itemVector[item, :] - self.userRegularizationFactor * self.userVector[user, :])
                self.itemVector[item, :] += self.learningRate * (error * self.userVector[user, :] - self.itemRegularizationFactor * self.itemVector[item, :])
             
    def get_mse(self, pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual) 

    def findAll(self):
        values = np.zeros((self.userVector.shape[0], self.itemVector.shape[0]))
        for u in xrange(self.userVector.shape[0]):
            for i in xrange(self.itemVector.shape[0]):
                val = self.globalBias + self.userBias[u] + self.itemBias[i]
                val += self.userVector[u, :].dot(self.itemVector[i, :].T)
                values[u, i] = val
        return values

    def train_test_split(self, ratings):
        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in xrange(ratings.shape[0]):
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=10, replace=False)

        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

        return train, test

    def calculateLearningCurve(self, iterArray, learningRate=0.1):
        iterArray.sort()
        trainMSE = []
        testMSE = []
        iter = 0
        
        train, test = self.train_test_split(self.ratingMatrix)
        
        for(i, iterN) in enumerate(iterArray):
            print ('Iteration: {}'.format(iterN))
            if i == 0:
                self.training(iterN - iter, learningRate)
            else:
                self.trainingStep(iterN - iter)

        findings = self.findAll()
        
        pred_max, pred_min = findings.max(), findings.min()
        findings = (findings - pred_min) / (pred_max - pred_min)

        for u in xrange(self.numberOfUsers):
            for i in xrange(self.numberOfItems):
                findings[u, i] = round(findings[u, i] * 5, 0)
        
        """for index1 in xrange(self.numberOfUsers):
            for index2 in xrange(self.numberOfItems):
                if findings[index1][index2] > 4.1:
                    print('{:4}'.format(self.ratingMatrix[index1][index2]))
                    print('{:4}'.format(findings[index1][index2]))
                    print(' ')"""
        
        trainMSE += [self.get_mse(findings, train)]
        testMSE += [self.get_mse(findings, test)]
        
        print ('Train mse: ' + str(trainMSE[-1]))
        print ('Test mse: ' + str(testMSE[-1]))
        
        iter = iterN
        
        hist = np.histogram(findings.flatten() , bins=[0,1,2,3,4,5,6])
        print(hist)
        
s = SGDC(40, 0.3, 0.3, 0.3, 0.3)
