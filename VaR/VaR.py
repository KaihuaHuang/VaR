# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:24:32 2018

@author: William Huang
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

class ValueAtRisk:
	def __init__(self,interval,matrix,weights):
		# Initialize the basic parameters
		# ----Input-----
		# interval: significant interval in statistic, range from 0 to 1
		# matrix: stock price matrix, each row represents one day price for different tickers, two dimentions ndarray
		# weight: the weight for portfolio, one dimension array
		# ----output----
		if(interval > 0 and interval < 1):
			self.ci = interval
		else:
			raise Exception("Invalid confidence interval", interval)

		if(isinstance(matrix,pd.DataFrame)):
			matrix = matrix.values

		if(matrix.ndim!=2):
			raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

		if(len(weights)!= matrix.shape[1]):
			raise Exception("Weights Length doesn't match")

		self.input = matrix
		# simple return calculation
		#self.returnMatrix = np.diff(self.input,axis = 0)/self.input[1:]

		# log return calculation
		self.returnMatrix = np.diff(np.log(self.input),axis=0)
		if (not isinstance(weights, np.ndarray)):
			self.weights = np.array(weights)
		else:
			self.weights = weights

	def covMatrix(self):
		# return variance-covariance matrix using return matrix
		# ----Input-----
		# interval: significant interval in statistic, range from 0 to 1
		# matrix: stock price matrix, each row represents one day price for different tickers, two dimentions ndarray
		# weight: the weight for portfolio, one dimension array
		# ----output----
		# variance-covariance matrix
		return np.cov(self.returnMatrix.T)

	def calculateVariance(self, Approximation = False):
		# return variance
		# ----Input-----
		# Approximation: If true, using portfolio return to calculate variance. If false, using cov-var matrix to calculate
		# ----output----
		# portfolio variance
		if(Approximation == True):
			self.variance = np.var(np.dot(self.returnMatrix,self.weights))
		else:
			self.variance = np.dot(np.dot(self.weights,np.cov(self.returnMatrix.T)),self.weights.T)
		return self.variance


	def var(self,marketValue = 0,Approximation = False,window = 252):
		# return parametric value at risk, the variance can be calculated by either cov matrix way or approximate way, scale the one day VaR according to user specified time period
		# ----Input-----
		# marketValue: the market value of portfolio, if the value is less or equal zero, function will return percentage result
		# approximation:  If true, using portfolio return to calculate variance. If false, using cov-var matrix to calculate
		# window: scale time period, default value is 252 which returns annualized VaR
		# ----output----
		# Value at Risk in dollar or percentage if input market value is zero
		if(self.returnMatrix.shape[1] != len(self.weights)):
			raise Exception("The weights and portfolio doesn't match")
		self.calculateVariance(Approximation)
		if(marketValue <= 0):
			return abs(norm.ppf(self.ci)*np.sqrt(self.variance))*math.sqrt(window)
		else:
			return abs(norm.ppf(self.ci)*np.sqrt(self.variance))*marketValue*math.sqrt(window)

	def setCI(self,interval):
		# set the confidence interval for value at risk
		# ----Input-----
		# interval: significant interval in statistic, range from 0 to 1
		# ----output----
		if(interval > 0 and interval < 1):
			self.ci = interval
		else:
			raise Exception("Invalid confidence interval", interval)

	def setPortfolio(self,matrix):
		# Change the current portfolio's data and weights
		# ----Input-----
		# matrix: stock price matrix, each row represents one day price for different tickers, two dimensions ndarray
		# ----output----
		if (isinstance(matrix, pd.DataFrame)):
			matrix = matrix.values

		if (matrix.ndim != 2):
			raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

		self.input = matrix
		self.returnMatrix = np.diff(np.log(self.input), axis=0)

	def setWeights(self,weights):
		# set the weights for the portfolio
		# ----Input-----
		# interval: the weight for portfolio, one dimension array
		# ----output----
		if (not isinstance(weights, np.ndarray)):
			self.weights = np.array(weights)
		else:
			self.weights = weights