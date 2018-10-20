# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:35:03 2018

@author: William Huang
"""
from VaR import ValueAtRisk
import numpy as np

class HistoricalVaR(ValueAtRisk):
	def var(self,marketValue=0,window = 0):
		# return historical VaR
		# ----Input-----
		# marketValue: the market value of portfolio, if the value less or equal zero, function will return percentage
		# window: look back period, if window is zero, it will use whole input price series
		# ----output----
		# Value at Risk in dollar or percentage if input market value is lee or equal zero
		self.portfolioReturn = np.dot(self.returnMatrix,self.weights)
		if(window >len(self.portfolioReturn)+1 ):
			raise  Exception("invalid Window, cannot excess", len(self.portfolioReturn))

		if(window > 0 and window < len(self.portfolioReturn)):
			PercentageVaR = abs(np.percentile(self.portfolioReturn[-window:],100*(1-self.ci),interpolation = 'nearest'))
		else:
			PercentageVaR = abs(np.percentile(self.portfolioReturn,100*(1-self.ci),interpolation = 'nearest'))

		if(marketValue <= 0):
			return PercentageVaR
		else:
			return PercentageVaR * marketValue
