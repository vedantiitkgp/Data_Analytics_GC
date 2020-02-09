#!/usr/bin/env python
# coding: utf-8

#Importing libraries
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from pylab import rcParams
from plotly import tools
#import plotly.plotly as py
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import math
from sklearn.metrics import mean_squared_error
#print(os.listdir("../input"))

##Changing these variables
index_reader = 5
no_of_epochs = 35

dataframe0 = pd.read_csv("./train.csv")
#dataframe1 = pd.read_csv('./building_1_anolomy.csv')
dataframe1 = pd.read_csv('./building_'+str(index_reader)+'_anolomy.csv')
#dataframe1 = pd.read_csv('./building_3_anolomy.csv')
#dataframe4 = pd.read_csv('./building_4_anolomy.csv')
#dataframe5 = pd.read_csv('./building_5_anolomy.csv')
shape_old_df1 = dataframe1.shape[0]
#Adding testfile 
test_new = pd.read_csv('./test_b'+str(index_reader)+'.csv')
#test_for_index = pd.read_csv('./test.csv')
#test_for_index['timestamp'] = pd.to_datetime(test_for_index['timestamp'],format='%Y-%m-%d %H:%M'
dataframe1 = pd.concat([dataframe1,test_new],axis=0)

interval = [dataframe1.shape[0]]#, dataframe2.shape[0], dataframe3.shape[0], dataframe4.shape[0], dataframe5.shape[0]]

dataframe = dataframe1#pd.concat([dataframe1, dataframe2, dataframe3, dataframe4, dataframe5], axis = 0)
print("The shape of raw data is -: ", dataframe.shape)
print("Shape of training data -: ", shape_old_df1)
print("Shape of testing data -: ", test_new.shape)

dataframe = dataframe.drop(['Unnamed: 0'],axis=1)
print(dataframe.info())
dataframe['timestamp'][:-test_new.shape[0]]=pd.to_datetime(dataframe['timestamp'][:-test_new.shape[0]],format='%d-%m-%Y %H:%M')
dataframe['timestamp'][-test_new.shape[0]:]=pd.to_datetime(dataframe['timestamp'][-test_new.shape[0]:],format='%d/%m/%Y %H:%M')
dataframe0['timestamp']=pd.to_datetime(dataframe0['timestamp'],format='%Y-%m-%d %H:%M:%S')


dataframe.set_index(dataframe['timestamp'],inplace=True)
dataframe = dataframe.drop(['timestamp'],axis=1)
dataframe.reset_index(inplace=True)

dataframe0.set_index(dataframe0['timestamp'],inplace=True)
dataframe0 = dataframe0.drop(['timestamp'],axis=1)
dataframe0.reset_index(inplace=True)

dataframe0.index = dataframe0['timestamp']
dataframe.index = dataframe['timestamp']
time_series = dataframe[['main_meter', 'sub_meter_1', 'sub_meter_2', 'corporate', 'day of week_0','day of week_1','day of week_2','day of week_3','day of week_4','day of week_5','day of week_6','building_number_1','building_number_2','building_number_3','building_number_4','building_number_5']] #'building_number','Hour', 'day of week']]#

dummy_test_data = dataframe0[['main_meter','sub_meter_1', 'sub_meter_2']]



from numpy import newaxis
from keras.layers.core import Dense ,Activation,Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

## LSTM

# In[67]:


##LSTM forecasting for the data

val_time = 1
n_input = 72
n_features = 16
val_length = test_new.shape[0]#int(float(dataframe1.shape[0])*0.2)
n_pred = 3
val_length_new = int(float(shape_old_df1)*0.2)
val_gap = val_time*4

sum = 0
#splitting into train and test data
for i, c in enumerate(interval):
	print(i)
	if (i == 0):
		train_data = time_series[:c-val_length]
		test_data = time_series[c-val_length:c]
		test_data_old = time_series[c-val_length-val_length_new:c-val_length]
		actual_test_data = dummy_test_data[(c-val_length-val_length_new)*val_gap:(c-val_length)*val_gap]
	else:
		train_data = pd.concat([train_data, time_series[sum:sum+c-val_length]])
		test_data = pd.concat([test_data, time_series[sum+c-val_length:sum+c]])
		# test_data_old = pd.concat([test_data_old, time_series[sum+c-]])
		actual_test_data = pd.concat([actual_test_data, dummy_test_data[(sum+c-val_length-val_length_new)*val_gap:(sum+c-val_length)*val_gap]])
	sum += c
	interval[i] = sum - (i+1)*val_length

test_data_old = pd.DataFrame(test_data_old).values.reshape(-1, n_features)
test_data = pd.DataFrame(test_data).values.reshape(-1,n_features)
train_data = pd.DataFrame(train_data).values.reshape(-1,n_features)
df_test = actual_test_data.copy()
actual_test_data = actual_test_data.to_numpy()
actual_test_data = actual_test_data.reshape(-1, n_pred)

print("Training data shape ", train_data.shape)
print("Testing data shape ", test_data.shape)
print("Testing data for evaluation shape ", test_data_old.shape)
print("Labels for evaluation shape (shold be 4 times) -", actual_test_data.shape)

#print(train_data.shape)
#starting with intialization 
scaler = MinMaxScaler()

scaler.fit(time_series[:interval[0]])
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data_old)


from keras.preprocessing.sequence import TimeseriesGenerator

X_train = []
y_train = []

for j, c in enumerate(interval):
	if (j==0):
		# print(n_input)
		for i in range(n_input, c):
			X_train.append(scaled_train_data[i-n_input:i, :])
			y_train.append(scaled_train_data[i, :n_pred])
	else:
		# print(interval[j-1]+n_input)
		for i in range(interval[j-1]+n_input, c):
			X_train.append(scaled_train_data[i-n_input:i, :])
			y_train.append(scaled_train_data[i, :n_pred])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], n_input, n_features))

perm = np.random.permutation(X_train.shape[0])
X_train = X_train[perm]
y_train = y_train[perm]

print("Shape of training data after final processing -: ", X_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD


val = "building"+str(index_reader)+"_1hr_sigmoid_3meter_adam_1_final"

lstm_model = Sequential()
lstm_model.add(GRU(300,  return_sequences=True, activation='relu',input_shape=(n_input,n_features)))
#lstm_model.add(GRU(50, return_sequences=True, activation='relu'))
lstm_model.add(GRU(50,activation='relu'))
lstm_model.add(Dense(3, activation='sigmoid'))

# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
lstm_model.compile(optimizer='adam',loss='mse')

print(lstm_model.summary())

model_json = lstm_model.to_json()
with open("./Json/model_LSTM_model_all_"+ val +".json", "w") as json_file:
    json_file.write(model_json)


lstm_model.fit(X_train, y_train, epochs=no_of_epochs, batch_size = 1024)

lstm_model.save_weights("./weights/model_LSTM_model_all_"+ val +".h5")
print("Saved model to disk")


json_file = open("./Json/model_LSTM_model_all_"+ val +".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
# load weights into new model
lstm_model.load_weights("./weights/model_LSTM_model_all_"+ val +".h5")
print("Loaded model from disk")

lstm_predictions_scaled = list()

sum = 0


for c in interval:
	batch = scaled_train_data[c-val_length_new-n_input:c-val_length_new, :]
	current_batch = batch.reshape((1,n_input,n_features))
	for i in range(sum, sum+val_length_new):   
		# print((lstm_model.predict(current_batch)).astype(np.float64))
		lstm_pred = ((lstm_model.predict(current_batch)).astype(np.float64))
		# print(train_data[c-n_input+i-sum,1], test_data[i][1])
		#print(lstm_pred, scaled_test_data[i][0], test_data[i][0])
		dummy = test_data_old[i].copy()
		dummy[:n_pred] = lstm_pred
		lstm_predictions_scaled.append(dummy)
		current_batch = np.append(current_batch[:,1:],[dummy.reshape(1, -1)], axis=1)
	sum += val_length_new


# for c in interval:
# 	batch = scaled_train_data[c-val_length_new-n_input:c-val_length_new, :]
# 	current_batch = batch.reshape((1,n_input,n_features))
# 	for i in range(sum, sum+val_length):   
# 		# print((lstm_model.predict(current_batch)).astype(np.float64))
# 		lstm_pred = ((lstm_model.predict(current_batch)).astype(np.float64))
# 		# print(train_data[c-n_input+i-sum,1], test_data[i][1])
# 		#print(lstm_pred, scaled_test_data[i][0], test_data[i][0])
# 		dummy = test_data[i].copy()
# 		dummy[:n_pred] = lstm_pred
# 		lstm_predictions_scaled.append(dummy)
# 		current_batch = np.append(current_batch[:,1:],[dummy.reshape(1, -1)], axis=1)
# 	sum += val_length

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
pred = np.array(lstm_predictions[:, :n_pred])
pred = np.repeat(pred, val_gap, axis = 0)
print(pred.shape)
# exit()
print(actual_test_data.shape)
pred = pred.reshape(-1, n_pred)
# for i in range(pred.shape[0]):
# 	print(pred[i][0], actual_test_data[i][0])
print(pred.shape, actual_test_data.shape)


def evalution_metric(m,m_hat,timestamp):
    Dt=timestamp.day
    Dt = Dt.to_numpy()
    # Dt = np.repeat(Dt, val_gap)
    print(Dt.shape)
    Sum=0
    for i in range(len(m)):
        Sum+=np.power((m[i]-m_hat[i]),2)*np.exp(-(np.log(2)/100)*Dt[i])
    score=(1/np.mean(m))*(np.sqrt(Sum))
    return score

# #calculating metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

sum = 0
for j, c in enumerate(interval):
	print("########################### For Building "+str(j)+"###########################")	
	for i in range(n_pred):
		mean_squared_error_lstm = mse(actual_test_data[sum:sum+val_length_new,i], pred[sum:sum+val_length_new,i])
		# mean_squared_log_error_lstm = msle(test_data,lstm_predictions)
		print("METER "+str(i+1)+"-: ")
		print(mean_squared_error_lstm**0.5)
		print(evalution_metric(actual_test_data[sum:sum+val_length_new,i],pred[sum:sum+val_length_new, i],df_test[sum:sum+val_length_new].index))
	sum += val_length


print(interval[0], shape_old_df1)
# assert interval[0] == shape_old_df1, "Shape of C do not match complete training data"

sum=0
lstm_predictions_scaled = list()

for c in interval:
	batch = scaled_train_data[c-n_input:c, :]
	current_batch = batch.reshape((1,n_input,n_features))
	for i in range(sum, sum+val_length):   
		# print((lstm_model.predict(current_batch)).astype(np.float64))
		lstm_pred = ((lstm_model.predict(current_batch)).astype(np.float64))
		#print(lstm_pred)
		#print(lstm_pred, scaled_test_data[i][0], test_data[i][0])
		dummy = test_data[i].copy()
		dummy[:n_pred] = lstm_pred
		lstm_predictions_scaled.append(dummy)
		current_batch = np.append(current_batch[:,1:],[dummy.reshape(1, -1)], axis=1)
	sum += val_length


lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
pred = np.array(lstm_predictions[:, :n_pred])
pred = np.repeat(pred, val_gap, axis = 0)

pred_dataframe = pd.DataFrame(pred).to_csv('./building_'+str(index_reader)+'_predictions.csv')



# # ## GRU

# # In[78]:


# from keras.layers import GRU
# gru_model = Sequential()
# gru_model.add(GRU(300,activation='relu',input_shape=(n_input,n_features)))
# gru_model.add(Dense(1))
# gru_model.compile(optimizer='adam',loss='mse')
# gru_model.summary()


# # In[79]:


# gru_model.fit_generator(generator,epochs=30)


# # In[549]:


# losses_gru = gru_model.history.history['loss']
# plt.figure(figsize=(12,4))
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks(np.arange(0,21,1))
# plt.plot(range(len(losses_gru)),losses_gru);


# # In[81]:


# gru_predictions_scaled = list()
# batch = scaled_train_data[-n_input:]
# current_batch = batch.reshape((1,n_input,n_features))

# for i in range(len(test_data)):   
#     gru_pred = gru_model.predict(current_batch)[0]
#     gru_predictions_scaled.append(gru_pred) 
#     current_batch = np.append(current_batch[:,1:,:],[[gru_pred]],axis=1)


# # In[82]:


# gru_predictions = scaler.inverse_transform(gru_predictions_scaled)
# gru_predictions = pd.Series(gru_predictions.reshape(1,-1)[0])
# gru_predictions.index = time_series[-40:].index


# # In[83]:


# ##Visualizing predictions by GRU
# plt.figure(figsize=(16,8))
# plt.plot(time_series[-50:],label='main_meter')
# plt.plot(gru_predictions,marker='o',label='GRU predictions')
# plt.title('Time Series Main meter')
# plt.legend(loc='best')


# # In[84]:


# #calculating metrics
# mean_squared_error_gru = mse(test_data,gru_predictions)
# mean_squared_log_error_gru = msle(test_data,gru_predictions)


# # ## Auto Arima

# # In[14]:


# from pmdarima.arima import auto_arima


# # In[ ]:


# stepwise_model =auto_arima(time_series[:-40], start_p=0, start_q=0,start_P=0,start_Q=0,
#                            max_p=4, max_q=4,max_P=3,max_Q=3,m=7,
#                            seasonal=True,d=1,
#                            D=1,trace=True,
#                            error_action='ignore',  
#                            suppress_warnings=True, 
#                            stepwise=False,random_state=0)


# # In[ ]:


# stepwise_model.fit(time_series[:-40])
# arima_predictions = stepwise_model.predict(n_periods=40)
# arima_predictions = pd.Series(arima_predictions)
# arima_predictions.index = time_series[-40:].index


# # In[ ]:


# ##Visualizing predictions by arima
# plt.figure(figsize=(16,8))
# plt.plot(time_series[-50:],label='main_meter')
# plt.plot(arima_predictions,marker='o',label='Auto arima predictions')
# plt.title('Time Series Main meter')
# plt.legend(loc='best')


# # In[ ]:


# #calculating metrics
# mean_squared_error_arima = mse(test_data,arima_predictions)
# mean_squared_log_error_arima = msle(test_data,arima_predictions)


# # ## TBATS

# # In[ ]:


# from tbats import TBATS
# estimator = TBATS(seasonal_periods=(0.5,7))
# model = estimator.fit(time_series[:-40])
# tbats_predictions = model.forecast(steps=40)


# # In[ ]:


# tbats_predictions = pd.Series(tbats_predictions)
# tbats_predictions.index = time_series[-40:].index


# # In[ ]:


# ##Visualizing predictions by arima
# plt.figure(figsize=(16,8))
# plt.plot(time_series[-50:],label='main_meter')
# plt.plot(tbats_predictions,marker='o',label='Tbats predictions')
# plt.title('Time Series Main meter')
# plt.legend(loc='best')


# # In[ ]:


# #calculating metrics
# mean_squared_error_tbats = mse(test_data,tbats_predictions)
# mean_squared_log_error_tbats = msle(test_data,tbats_predictions)


# # ## Prophet

# # In[ ]:


# ts = pd.DataFrame(time_series)
# ts = ts.reset_index(drop=True)
# ts = pd.concat([pd.DataFrame(time_series.index),ts],axis=1)
# ts.columns = ['ds','y'] # To use prophet column names should be like that 
# train_data_pr = ts.iloc[:len(time_series)-40]
# test_data_pr = ts.iloc[len(time_series)-40:]
# from fbprophet import Prophet
# model = Prophet()
# model.fit(train_data_pr)
# future = model.make_future_dataframe(periods=40,freq='D')
# prophet_predictions = model.predict(future)


# # In[ ]:


# prophet_predictions = pd.DataFrame({"Date" : prophet_predictions[-40:]['ds'], "Pred" : prophet_predictions[-40:]["yhat"]})
# prophet_predictions = prophet_predictions.set_index("Date")


# # In[ ]:


# prophet_predictions = prophet_predictions.iloc[:,0]


# # In[ ]:


# prophet_predictions.index = time_series[-40:].index


# # In[ ]:


# ##Visualizing predictions by arima
# plt.figure(figsize=(16,8))
# plt.plot(time_series[-50:],label='main_meter')
# plt.plot(prophet_predictions,marker='o',label='Prophet predictions')
# plt.title('Time Series Main meter')
# plt.legend(loc='best')


# # In[ ]:


# #calculating metrics
# mean_squared_error_prophet = mse(test_data,prophet_predictions)
# mean_squared_log_error_prophet = msle(test_data,prophet_predictions)


# # In[ ]:


# mean_squared_errors =[mean_squared_error_lstm,mean_squared_error_gru,mean_squared_error_arima,mean_squared_error_tbats,mean_squared_error_prophet]
# mean_squared_log_errors =[mean_squared_log_error_lstm,mean_squared_log_error_gru,mean_squared_log_error_arima,mean_squared_log_error_tbats,mean_squared_log_error_prophet]
# mean_errors = pd.concat([pd.DataFrame(mean_squared_errors),pd.DataFrame(mean_squared_log_errors)],axis=1)
# mean_errors.columns = ['mean_squared_error','mean_squared_log_error']
# mean_errors.index = ['LSTM','GRU','Arima','Tbats','Prophet']
