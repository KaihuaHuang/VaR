# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:05:57 2018

@author: William Huang
"""

from VaR import ValueAtRisk
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pandas as pd
import math


class PCAVaR(ValueAtRisk):
    def __init__(self,interval,matrix,universe,weights = np.ones((1))):
        # Initialize the parameters
        # ----Input-----
        # interval: significant interval in statistic, range from 0 to 1
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimensions ndarray
        # universe: the stock universe to generate PCA components
        # weight: the weight for portfolio, one dimension ndarray, default value is 1 which means there is only 1 stock in portfolio
        # ----output----
        if(len(matrix)!=len(universe)):
            raise Exception('The length of input data and the length of universe data should match')
        ValueAtRisk.__init__(self,interval,matrix,weights)
        if(isinstance(universe,pd.DataFrame)):
            universe = universe.values
        self.universe = universe
        self.universeReturnMatrix = np.diff(np.log(self.universe),axis=0)
    
    def getComponents(self,n_components = 2):
        # Generate principle components
        # ----Input-----
        # n_components: the number of components user want to generate
        # ----output----
        # factor matrix
        if(self.universe.shape[1]<n_components):
            raise Exception("Too many PCA Components")
        pca = PCA(n_components=n_components)
        pca.fit(self.universeReturnMatrix)
        self.betaMatrix = pca.components_
        self.factorMatrix = np.dot(self.universeReturnMatrix,self.betaMatrix.T)
        self.factorCovVarMat = np.cov(self.factorMatrix.T)
        return self.factorMatrix


    def betaRegression(self,returns):
        # Run linear regression on return series
        # ----Input-----
        # returns: return series, the return's date should match factor's date
        #          eg. the first return and the first row factors are in the same date
        # ----output----
        # regression coefficient
        reg = LinearRegression().fit(self.factorMatrix, returns)
        self.betaMatrix = reg.coef_.T
        return reg.coef_.T


    def varSingle(self,prices,marketValue = 0,window = 252):
        # return Value at risk for single price series
        # ----Input-----
        # n_components: the number of components user want to generate
        # ----output----
        # factor matrix
        if(isinstance(prices,np.ndarray)):
            prices = np.reshape(prices,(-1,1))
            returns = np.diff(prices, axis=0) / prices[1:]
        else:
            returns = np.diff(prices.values, axis=0) / prices.values[1:]
        self.betaRegression(returns)
        self.CovVarMat = np.dot(np.dot(self.betaMatrix.T, self.factorCovVarMat), self.betaMatrix)
        self.variance = self.CovVarMat[0,0]
        if (marketValue <= 0):
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)
        else:
            return marketValue * abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)


    def changePortfolio(self,matrix,weights):
        # Change the current portfolio's data and weights
        # ----Input-----
        # matrix: stock price matrix, each row represents one day price for different tickers, two dimensions ndarray
        # weight: the weight for portfolio, one dimension ndarray, default value is 1 which means there is only 1 stock in portfolio
        # ----output----
        ValueAtRisk.__init__(self, self.ci, matrix, weights)


    def var(self,marketValue = 0,window = 252,approximation = False):
        # Return value at risk for portfolio
        # ----Input-----
        # marketValue: the market value of portfolio, if the value is less or equal zero, function will return percentage result
        # approximation:  If true, using portfolio return to run beta regression. If false, using each stock series to run beta regression
        # window: scale time period, default value is 252 which returns annualized VaR
        # ----output----
        # Value at Risk in dollar or percentage if input market value is zero 
        if (approximation):
            portfolioReturn = np.dot(self.input, self.weights)
            return self.varSingle(portfolioReturn, marketValue, window)
        else:
            colNum = self.input.shape[1]
            betas = []
            for i in range(colNum):
                singlePrice = self.input[:, i]
                singleReturn = np.diff(singlePrice, axis=0) / singlePrice[1:]
                betas.append(list(self.betaRegression(singleReturn)))
            self.betaMatrix = np.array(betas).T
            self.CovVarMat = np.dot(np.dot(self.betaMatrix.T, self.factorCovVarMat), self.betaMatrix)
            self.variance = np.dot(np.dot(self.weights, self.CovVarMat), self.weights.T)
            if (marketValue <= 0):
                return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)
            else:
                return marketValue * abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(window)


