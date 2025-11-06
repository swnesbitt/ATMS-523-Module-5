#Summary
This notebook sets up a 70/30 training/testing multi-linear regression model to predict total rainfall using radar-derived parameters. The Marshall-Palmer relationship is used as a baseline prediction.
This equation involves the radar reflectivity factor which can be solved by using dBZ. Additional model runs are done using polynomial/cross-validation models, as well as a RandomForestRegressor. 

#References
Chase, R. J., D. R. Harrison, A. Burke, G. M. Lackmann, and A. McGovern, 2022: A Machine Learning Tutorial for Operational Meteorology. Part I: Traditional Machine Learning. Wea. Forecasting, 37, 1509–1529, https://doi.org/10.1175/WAF-D-22-0070.1

Chase, R. J., D. R. Harrison, G. M. Lackmann, and A. McGovern, 2023: A Machine Learning Tutorial for Operational Meteorology. Part II: Neural Networks and Deep Learning. Wea. Forecasting, 38, 1271–1293, https://doi.org/10.1175/WAF-D-22-0187.1

Marshall, J. S., and Palmer, W. M., 1948: The Distributions of Raindrops with Size. Journal of Meteorology, 5, 165-166, https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2 
