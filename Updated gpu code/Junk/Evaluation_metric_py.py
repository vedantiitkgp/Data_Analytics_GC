
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from datetime import datetime    # To access datetime 
from pandas import Series 
import warnings
# warnings.filterwarnings('ignore')


# In[12]:


data=pd.read_excel('./Building_data/Evaluation Metric example.xlsx')


# In[13]:


data.head()


# In[14]:


timestamp=data['timestamp']
timestamp=timestamp[:192]


# In[15]:


m=data['m1']
m=m[:192]
m_hat=data['m1_hat']
m_hat=m_hat[:192]
m=m.to_numpy()
m_hat=m_hat.to_numpy()


# In[16]:


# timestamp is a series of timestamp values
# m is the numpy array of actual value of a meter
# m_hat is the numpy array of predicted values of a meter
def evalution_metric(m,m_hat,timestamp):
    Dt=timestamp.dt.day
    print(Dt.shape)
    print(Dt[95])
    Dt = Dt.to_numpy()
    Dt = np.repeat(Dt, 4)
    print(Dt.shape)
    print(Dt[384])
    Sum=0
    # for i in range(len(m)):
       	# print(Dt[i])
    #     Sum+=np.power((m[i]-m_hat[i]),2)*np.exp(-(np.log(2)/100)*Dt[i])
    # score=(1/np.mean(m))*(np.sqrt(Sum))
    return Sum


# In[17]:


value=evalution_metric(m,m_hat,timestamp)

# print(value)

