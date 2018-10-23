# VaR
VaR calculation python library, including historical VaR, parametric VaR and PCA VaR  
Current Version: v1.0  
Version Released: 10/20/2018  
Third-Party Dependency: numpy, pandas, scipy, sklearn  
Version Requirements: dateutil 2.7.3  
Report any bugs by opening an issue here: https://github.com/KaihuaHuang/VaR/issues  
  
## Input data requirements
### Historical VaR
The data should ascend by date which means the last row is most up-to-date  
  
### PCA VaR
The date of portfolio date should match the date of universe data  
For example, the first row of portfolio price data and universe price data comes from the same day. This part can be done by my another library FinanceData  
  
## Demo Data
### universe.csv  
Price data from top 20 market capitalization stocks in 11 GIC sectors which in total 220 stocks. The date of price ranges from 2017/10/09 to 2018/10/08. This data set is used for generate principle components.  
  
### singleStock.csv
The price data from AAR Corp. (AIR). The date of price ranges from 2017/10/09 to 2018/10/08 which matches the dates in universe.csv

### portfolio.csv  
The portfolio price data from 2017/10/09 to 2018/10/08. The portfolio includes AIR, MMM, DIS and UPS.



