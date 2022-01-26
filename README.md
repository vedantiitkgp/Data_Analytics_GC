# Electricity Demand Forecast - Data Analytics GC (RP Silver ) 

![](https://www.deccanherald.com/sites/dh/files/styles/article_detail/public/articleimages/2020/07/25/electricity%20istock-1595676921.jpg?itok=TTPLAi38)

Electrical Energy is one of the most important sources for the social and economic development of all nations. The growth in energy consumption is essentially linked with the growth in the economy. Building energy consumption has become one of the three major energy-consuming industries, in addition to the industrial and transportation industries.

The ability to forecast electricity demand accurately is one of the most important tasks for power system planning.Load prediction helps in reliable and economical operation of the power system. In general, if the forecasted load is greater than the demand there will be an unnecessary commitment of units, whereas if the forecasted load is lower than the demand, it will lead to the purchase of power at higher prices in the deregulated market.

Hence, we aim to develop efficient time series models for Dream Vidyut and help them to better forecast electricity consumption of provided corporate buildings. We have used data visualization tools, to generate insights and developed intelligent features to support our forecasting models. Identification of anomalies was done, assignable causes were identified, and they are replaced with mean values to develop robust models. We have developed various statistical, machine learning and deep learning-based models, and evaluated them on the validation set for model selection. Finally, we stacked results from the two best performing models to produce robust predictions


## Exploratory Data Analysis

### Data Summary

#### Summary Statistics

The data provided consists of the reading in the main meter and two sub-meters of five buildings from 01-04-2017 to 31-12-2017. The given timestamp is divided into 15-minute intervals. The dataset consists of electricity consumption of five buildings across three different meters for each timestamp.

<img width="706" alt="Screenshot 2022-01-26 at 10 33 30 PM (1)" src="https://user-images.githubusercontent.com/32813089/151211445-6ff191d2-fde4-41de-8598-86ac88719b3d.png">

From the mean, 75% and the maximum readings of the meters, it can be seen that the dataset consists of outliers that have to be treated for better prediction. It can also be seen from the above table that 25% of the Sub Meter 2 readings are below 4.85, which suggests that electricity consumption through sub meter 2 is only during some portion of time

#### Correlation Matrices

We have calculated the correlation matrix across buildings for all the metres and found that the electricity consumptions between the corporate buildings are fairly correlated, which shows that the model can be generalized to other new corporate buildings too

<img width="686" alt="Screenshot 2022-01-26 at 10 37 03 PM" src="https://user-images.githubusercontent.com/32813089/151212110-29c01e49-fe07-460c-92a9-6f32e4ba4061.png">

<img width="722" alt="Screenshot 2022-01-26 at 10 40 32 PM" src="https://user-images.githubusercontent.com/32813089/151212358-44b76cec-5faf-4f87-8556-e4bf1ef7d1a1.png">

It is visible that meter readings are also fairly correlated with each other, which can be justified as the electricity consumption generally depends on the day-wise factors such as temperature, humidity et

### Data Visualisation and Insights

Data visualization is one of the first steps of building data-driven forecasting models. It is a well-known fact that the representation of data in terms of the graph, charts etc grabs our interest and helps us to internalize the patterns quickly

#### General Trend in Time Series

We first plotted a canvas of line charts of each time series, ranging over all the buildings, to get a brief overview of what kind of time series we have, and what further analysis we should perform. It is noticeable that the variance of readings of the main meter are considerably higher than that of the sub-meters. Also, it is visible that as compared to submeter 1, sub-meter 2 has fairly less variance. It is also clear that submeter 2 has higher consumption when compared to that
of submeter 1 and mainmeter

<img width="681" alt="Screenshot 2022-01-26 at 10 44 51 PM" src="https://user-images.githubusercontent.com/32813089/151213190-752dfe47-0446-408e-bf12-8b3198e18ae5.png">

#### Seasonality within a Week

We had an intuition that there should be a significant difference between the distribution of energy consumption during the weekdays as compared to weekends. Our intuitions were further strengthened by plotting the boxplots. 

<img width="567" alt="Screenshot 2022-01-26 at 10 44 56 PM" src="https://user-images.githubusercontent.com/32813089/151213460-4c642b09-7ade-4d40-a813-fe10bdbe3e70.png">

#### Seasonality within a Day

As the given building is a corporate building, an obvious question arises if the readings while the data from corporate & non-corporate hours (08:00-20:00) might follow a certain trend. We confirmed this intuitive idea by plotting the aggregated day-wise meter readings. It is clearly visible that there is a significant difference between the electricity consumption during corporate and non-corporate hours.

<img width="723" alt="Screenshot 2022-01-26 at 10 51 48 PM" src="https://user-images.githubusercontent.com/32813089/151214136-0da36a43-b961-41e3-a9d6-dbd0f02e0672.png">

## Feature Engineering

### Weekend Feature

We have identified through the above graphs that the distribution of energy consumption during the weekdays is very different from that of Weekends. So, we have made a new feature called Weekend which has a value of 1 on weekends and a value of 0 on weekdays

<img width="740" alt="Screenshot 2022-01-26 at 10 57 00 PM (1)" src="https://user-images.githubusercontent.com/32813089/151215068-06b1027e-b2d0-4cfc-b1bc-23edd6e409f9.png">

Confidence Level = 0.95, Degree of Freedom = 32998
The null hypothesis is rejected and the time series are significantly different for the added weekend feature.

### Corporate Feature

As we know, the meters have significantly different readings during corporate hours (8 AM to 8 PM). So, a new feature called corporate was made which had a value of 1 during corporate hours and a value of 0 during non-corporate hours

<img width="760" alt="Screenshot 2022-01-26 at 10 59 46 PM (1)" src="https://user-images.githubusercontent.com/32813089/151215680-f474cd67-bea5-4559-b830-98c82fed05a7.png">

Confidence Level = 0.95, Degree of Freedom = 32998
The null hypothesis is rejected and the time series is significantly different for the added corporate feature.

## Anomaly Detection

<img width="565" alt="Screenshot 2022-01-26 at 11 03 56 PM" src="https://user-images.githubusercontent.com/32813089/151216193-045d140a-04f0-4b0a-b38d-ca09af776e3e.png">

As we can see from these plots, there are clear outliers, so we decided to replace these outliers with the median values of each of the series. 

## Models - Time Series Models

We applied various Time Series models from different classes such as Statistical (Holtz-Winter’s Method, ARMA, ARIMA, Garch), Machine Learning based (XGBoost Regressor, Random Forest Regressor) as well as Deep Learning-based (RNNs, LSTMs, GRU) models. We have presented below the results for the best models from each class and our intuition of why these models performed the way they have performed.

### ARIMA

An ARIMA model is a class of statistical models for analyzing and forecasting equally spaced univariate time series data and also predicts a value in a response time series as a linear combination of its own past values, past errors and current and past values of other time series. ARIMA models allow both autoregressive (AR) components as well as moving average (MA) components

<img width="686" alt="Screenshot 2022-01-26 at 11 08 31 PM" src="https://user-images.githubusercontent.com/32813089/151222284-5e8449c7-e1b2-414c-b800-3a3c87005e08.png">

```
from pmdarima import auto_arima
stepwise_model = auto_arima(dataframe, start_p=1, start_q=1,max_p=5, max_q=5,start_P=0, seasonal=False,d=1, D=1, trace=True)

ARIMA(maxiter=50, method='lbfgs', order=(2, 1, 1), out_of_sample_size=0,
      scoring='mse', scoring_args=None, seasonal_order=(0, 0, 0, 0),
      start_params=None, suppress_warnings=True, trend=None,
      with_intercept=True)
 
```

### XG-Boost

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. It is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction

```
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=1000)
```

### GRU (Gated Recurrent Unit)

GRU is an updated version of RNN which solves the vanishing gradient problem of standard Recurrent Neural Network. GRUs remember the past sequence using forget gates and reset gates. GRUs is preferred over LSTMs.

<img width="711" alt="Screenshot 2022-01-26 at 11 49 58 PM (1)" src="https://user-images.githubusercontent.com/32813089/151223292-d73c086b-c190-4146-b581-52de0c5b0562.png">

```
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential

gru_model = Sequential()
gru_model.add(GRU(300,  return_sequences=True, activation='relu',input_shape=(n_input,n_features)))
gru_model.add(GRU(50,activation='relu'))
gru_model.add(Dense(3, activation='sigmoid'))
gru_model.compile(optimizer='adam',loss='mse')
```


## Result Analysis

GRU has a lower evaluation metric score for most of the buildings, then that of ARIMA and XGBoost. ARIMA has performed well in the meters where the variance is less, while GRU is able to learn more complex patterns in the dataset. XGBoost’s results are good among the other machine learning algorithms but its performance is not at par with other models which are tailor-made for sequential models.

<img width="542" alt="Screenshot 2022-01-26 at 11 57 01 PM" src="https://user-images.githubusercontent.com/32813089/151224550-69dafe13-1846-4f4c-9994-622d86c7b169.png">


### Stacked Predictions

Stacked predictions allow you to make “out-of-sample” predictions and prevent misleadingly high accuracy scores.
We are stacking the predictions from GRU & ARIMA Models, and we use the value of the evaluation metric for the model on the validation set to decide the weights. Weights are determined using the formulas below, and final predictions are given as a weighted average of predictions from each model.

<img width="614" alt="Screenshot 2022-01-27 at 12 01 51 AM" src="https://user-images.githubusercontent.com/32813089/151225126-c02abc0d-8208-41c0-a091-4514c3edc65e.png">

<img width="557" alt="Screenshot 2022-01-27 at 12 01 51 AM (1)" src="https://user-images.githubusercontent.com/32813089/151225904-568d8176-b400-44b7-b551-8a120f90251a.png">

