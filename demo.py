# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:43:27 2018

@author: William Huang
"""
from VaR import ValueAtRisk
from VaR import HistoricalVaR
from VaR import PCAVaR
import pandas as pd
import numpy as np

if __name__ =='__main__':
	print('--------Test Main----------')
	weights = np.array([0.1,0.3,0.4,0.2])
	data = pd.read_csv('Data/portfolio.csv', index_col='date', dtype=float, parse_dates=True)


	print('--------data preview--------')
	print(data.head())
	print('\n--------Parametric Var--------')
	Demo = ValueAtRisk(0.95,data,weights)
	print('Covariance Matrix\n',Demo.covMatrix())
	print('Annualized VaR(Percentage): ',Demo.var()*100,'%')
	print('Annualized VaR(Dollar): ',Demo.var(marketValue=1000000))
	print('Daily VaR(Percentage): ',Demo.var(window = 1)*100,'%')
	print('Daily VaR(Dollar): ',Demo.var(marketValue=1000000,window = 1))
	print('Annualized VaR(Percentage) - Approximation: ',Demo.var(Approximation = True)* 100,'%')
	print('Annualized VaR(Dollar) - Approximation: ',Demo.var(Approximation = True,marketValue=1000000))

	print('\n--------Historical Var--------')
	HDemo = HistoricalVaR(0.95,data.as_matrix(),weights)
	print('VaR(Percentage): ',HDemo.var(),'%')
	print('Var(Dollar):',HDemo.var(marketValue = 1000000))
	print('100 day - VaR(Percentage): ',HDemo.var(window = 100)*100,'%')
	print('100 day - Var(Dollar):',HDemo.var(marketValue = 1000000,window = 100))

	print('\n--------Single PCA VaR--------')
	universe = pd.read_csv('Data/universe.csv',index_col = 'date',dtype=float,parse_dates=True)
	singleStock = pd.read_csv('Data/singleStock.csv',index_col = 'date',dtype=float,parse_dates=True)
	PDemoValidation = PCAVaR(0.95,singleStock,universe)
	PDemoValidation.getComponents(3)
	print('Using 220 stocks as universe to generate 3 components')
	print('Single PCA VaR(Percentage):',PDemoValidation.var() * 100,'%')
	print('Single PCA VaR(Dollar):', PDemoValidation.var(marketValue = 1000000))

	print('\n-------Portfolio PCA VaR---------')
	data = pd.read_csv('Data/portfolio.csv', index_col='date', dtype=float, parse_dates=True)
	PDemoValidation.setPortfolio(data)
	PDemoValidation.setWeights(weights)
	print('Portfolio PCA VaR(Percentage):', PDemoValidation.var() * 100, '%')
	print('Portfolio PCA VaR(Percentage)-Approximation:', PDemoValidation.var(approximation=True) * 100, '%')

