#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from root_numpy import tree2array
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler  
from numpy import concatenate

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from math import *

import h5py

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams["image.origin"] = 'lower'


# In[5]:


inputfile = pd.read_csv("output_metadata_2017_10min.csv", sep=",")
true = np.load("output_transp_EB_ix1_10min.npy", mmap_mode = "r")


# In[ ]:


# Cleaning way first fill


# # hyperparameters
# - encoder: 16 samples of 40 minutes, 1 interval 10 min, 1 every 4  --> 64 samples tstart
# - decoder: 48 samples, 10 min == 8 hours
# 
# **Window of 114 samples**

# In[8]:


encod_inputs = []
decod_inputs_and_true = []


# In[9]:


window_step = 20
window_width = 114 
tstart = 63
steps_gen = 48
skip_t_encod = 4

tot = (len(inputfile)-window_width ) // window_step

for iev, t0 in enumerate(range(0, len(inputfile)-3*window_width, window_step)):
    if iev % 50 == 0:
        print(f"{iev}/{tot}> t0={t0}")
    
    t1 = t0+tstart
    t2 = t1+ steps_gen +1  # t2 excluded

    #print(t0,t1,t2)
    inputs = inputfile.iloc[t0:t2]

    
    inputs["deltaT"] = inputs["time"] - inputs["time"].iloc[t1-t0]

    encod_input_meta_all = inputs[inputs.deltaT<=0][["deltaT","in_fill","lumi_inst", "lumi_in_fill", "lumi_last_fill",
                                            "time_in_fill", "time_in_fill_stable", "time_from_last_fill", 
                                            "last_dump_duration", "last_fill_duration"]]

    encod_input_meta = encod_input_meta_all.values[::skip_t_encod,]

    decod_input = inputs[inputs.deltaT>0][["deltaT","in_fill","lumi_inst", "lumi_in_fill", "lumi_last_fill",
                                            "time_in_fill", "time_in_fill_stable", "time_from_last_fill", 
                                            "last_dump_duration", "last_fill_duration", "lumi_since_last_point"]].values

    transp = true[t0:t2].T

    for ixtal in range(transp.shape[0]):
        #print(transp.shape, t2-t0)
        transp_byxtal = transp[ixtal,:].reshape( t2-t0 ,1)
        # from t0 to t1 compreso every 4
        transp_encoder = transp_byxtal[0:t1-t0+1:skip_t_encod]  # include the tstart timestamp
        transp_decoder = transp_byxtal[t1-t0+1: t2-t0]

        #print(transp_byxtal.shape, t1+1, t2, transp_encoder.shape, transp_decoder.shape)

        encod_input_with_transp = np.hstack([encod_input_meta, transp_encoder])
        encod_inputs.append(encod_input_with_transp)

        decod_input_with_transp =  np.hstack([decod_input, transp_decoder])
        decod_inputs_and_true.append(decod_input_with_transp)

    
    


# In[ ]:


encod_inputs = np.array(encod_inputs)
decod_inputs_and_true = np.array(decod_inputs_and_true)


# In[ ]:


import h5py
datafile = h5py.File("ECAL4ML_datasets.hd5f", "w")


# In[ ]:


dset_input = datafile.create_dataset("encoder_input_v1", data=encod_inputs)
dset_gen = datafile.create_dataset("decoder_input_and_true_v1", data=decod_inputs_and_true)

