# Forecast_VOOStocks

![image](https://github.com/weibb123/Forecast_VOOStocks/assets/84426364/ec08b0e6-dfb5-4536-9b2d-4b3c100cba06)




## Table of Contents

  - [Business problem](#business-problem)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Evaluation](#evaluation)
  - [Reference](#reference)

    ## Business problem
    As an investor myself who invests in VOO for long term. Some desiring question I want to know is the future price of VOO.\
    The problem here to solve is to forecast the stocks of VOO to see its trejectory.

    ## Data source
    Retrieve data of VOO stocks from Yahoo Finance API. This webapp updates the VOO stock daily.\
    As for the size of the data, I grabbed 10 years worth of VOO stock's data to make forecast. I believe that more data will create a better forecasting graph and learn the pattern of the stock well enough.

    ## Methods
    To make good prediction of timeseries, we use prophet, timseries algorithm by Meta.

    Before using the model, it is important for us to split the dates to create train and test sets and transform the column to fit the algorithm.\
    Prophet expects the dataset to be in 2 columns, date and y. In this case, date will be the date of VOO stocks overtime, y will be the open price of VOO stocks.\
    Next, I want to split the dates into 80% trainset and 20% testset as a common practice in ML pipeline.\

    ## Tech Stack
    
    - Streamlit
    - Python
    - Prophet
    - Pandas/Seaborn/Matplotlib
    
    ## Lessons learned and recommendation
     Key lesson: forecasting stock is a difficult task, not that it is impossible.
      ![image](https://github.com/weibb123/Forecast_VOOStocks/assets/84426364/7c8fadef-9e71-4fa7-a1f6-ca209e82ab0e)
    
        By looking at the forecasting, we see that the trendline goes in a wave pattern. In some sense, it shows that VOO stock is a relatively stable stock. However, looking at the graph, the model didn't   
       expect the recessions or covid period. Hence, My takeway is that VOO stock will yield stable returns if theres exist no events that damage the US market.

      Forecasting stocks is the not only thing that prohpet algorithm can do. It can be used on forecasting sales and other timeseries data such as weather. Prophet algorithm by facebook can achieve high             accuracy on other timeseries data except stock data.
    
    ## Limitation and what can be improved
  
    Forecasting near future not deep future
    Forecasting deep into the future might not correspond to the actual stocking price. Instead, it might be a good approach to focus on forecasting the next couple days or weeks.

    Forecasting Prophet can have some drawbacks...

    What is good about Prophet:
    
    1. Good for data that has daily, weekly, yearly seasonality..
    
    2. robust to outliers

    3. relatively computationally efficient

    4. use it when data has large mean shift
    
    What is not good about Prophet:
    
    1. Below performance compared to ARIMA on situation where peak predictive performance is important
    
    2. only appropiate for univerate time series and cannot account multiple correlated time series


    ## Evaluation

      Metric used: Mean_absolute_error, and the reported mean_aboslute_error is...\
      ![image](https://github.com/weibb123/Forecast_VOOStocks/assets/84426364/a665049e-be29-43e6-88c5-f5142bbf2e1f)

      Looking at the predicted line and the actual stock price, the model does capture the majority of the datapoints well except the year 2020 global crisis and recessions.
    
    ## Reference
      - https://facebook.github.io/prophet/
  
