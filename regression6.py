#automate data preparation with few functions 
#baseline setup
from cmath import exp
from itertools import tee
from tkinter.ttk import LabelFrame
import matplotlib.pyplot as plt
#import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np

#libraries for general data analysis 
from pprint import pprint
from collections import namedtuple
import datetime
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import get_session

#libraries for DNN
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM  #might be usefull for further works
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import seaborn as sns

data_folder = ('/home/federico/root/root-6.24.06-install')
#data_folder = ('/gwpool/users/fdematteis') #CERN server
#dictionary_train for npy files containing rough transparency data


#function that creates dataset ready for the training 
def Pre_Processing_Train(dictionary_train,metadata):

    transp_train = []
    instLumi = []
    intLumiLHC = []
    infillLumi = []
    lastfillLumi = []
    lastpointLumi = []
    filltime = []
    true_time = []
    ring_index = []

    Norm_time_in_fill_train = []
    Norm_time_in_fill_prov_train = []
    weights_train = []
    FILL_num = []
    METADATA_train=[]

    for i_key, i_value in dictionary_train.items():
        print(i_value[0])

        data=np.load(f"{data_folder}/{i_value[0]}")
        data_df = pd.DataFrame(data)
        data_df.head()
    
        mean=[]
        for k in range (0, len(data_df.axes[1])):
            mean = np.append(mean, np.mean(data_df[k]))

        #mean transparency in iring
        mean=mean[mean != -1]
        metadata = metadata.iloc[:len(mean)][mean != -1]

        #selecting metadata for fill (locking metadata to in_fill=1)
        #fill arrays are lists of fill_num related to each crystal
        fill=metadata["fill_num"].unique()
        fill = fill[fill != 0]
        
        #excluding NoNSMooTh fills from the "fill" array
        #non va bene definire così i fill da escludere
        #devo far corrispondere a value di NoNSMooth value di training data
        NONsMOOTH=i_value[1]
        for iev in range (0, len(NONsMOOTH)) :
            fill = fill[fill != NONsMOOTH[iev]]

        metadata_fill = metadata[metadata.fill_num.isin(fill)]
        metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
        fill_num = metadata_fill.fill_num.unique()
        FILL_num = np.append(FILL_num, fill_num)
        METADATA_train=np.append(METADATA_train, metadata_fill)

        #Normalised time in fill for xtal 23 (iring23)
        for k in fill_num:
            dftrain = metadata_fill[metadata_fill.fill_num == k]
            #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
            Norm_time_in_fill_prov_train = dftrain.loc[:,'time_in_fill']
            Norm_time_in_fill_prov_train = Norm_time_in_fill_prov_train/dftrain['time_in_fill'].iloc[0]
            Norm_time_in_fill_train=np.append(Norm_time_in_fill_train,Norm_time_in_fill_prov_train)
    
    
        #MEAN TRANSPARENCY in iRing (The target function)
        transp_fill = []
        for k in fill_num:
            df = metadata_fill[metadata_fill.fill_num == k]
            transp = [mean[i] for i in df.index.values]
            transp = transp/transp[0]
            transp_fill = np.append(transp_fill, transp)

        transp_train = np.append(transp_train,transp_fill)
        #--------------------------------------TRAINING DATA------------------------------------------------------------
        #Metadata (input) for training related to each xtal
        #fill_num is different for subdataset related to each crystal,
        #it depends on which fills we excluded form the train

        instLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_inst']
        intLumiLHC_p = (1e-9)*metadata_fill.loc[:,'lumi_int']
        infillLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_in_fill']
        lastfillLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_last_fill']
        filltime_p = (1e-9)*metadata_fill.loc[:,'time_in_fill']
        lastpointLumi_p = (1e-9)*metadata_fill.loc[:, 'lumi_since_last_point']
        true_time_p = (1e-9)*metadata_fill.loc[:, 'time']
        ring_index_p = np.zeros(len(metadata_fill))
        print(i_key)
        for j in range (0, len(ring_index_p)):
            ring_index_p[j] = i_key
        
        
        #merge data into a single object
        instLumi=np.append(instLumi, instLumi_p)
        intLumiLHC=np.append(intLumiLHC, intLumiLHC_p)
        infillLumi=np.append(infillLumi, infillLumi_p)
        lastfillLumi=np.append(lastfillLumi, lastfillLumi_p)
        filltime=np.append(filltime, filltime_p)
        lastpointLumi=np.append(lastpointLumi, lastpointLumi_p)
        true_time=np.append(true_time, true_time_p)
        ring_index = np.append(ring_index, ring_index_p)


    #Virtual weights for samplle weighting 
    #len(FILL_num) is the number of first instances over all fills used for training, for various xtals.
    #len(Norm_time_in_fill) is the total number of instances for all fills, for various xtals.

    for k in range (0, len(Norm_time_in_fill_train)):
        if Norm_time_in_fill_train[k] == 1. :
            weight = 0.001/(len(FILL_num))
            weights_train.append(weight)
        else:
            #weight = (1-0.5)/ (len(fill_num2)+len(merged_norm_time_in_fill)-(len(fill_num)+len(fill_num1)))
            weight = (1-0.001)/(len(Norm_time_in_fill_train)-(len(FILL_num)))
            weights_train.append(weight)

        #merge metadata for each xtal into a single object
    all_inputs_train=np.stack((instLumi,intLumiLHC,infillLumi,lastfillLumi,filltime,lastpointLumi, ring_index), axis=-1)

    return [all_inputs_train, transp_train, weights_train]

#function that creates dataset ready for inference
def Pre_Processing_Test(dictionary_test,metadata):
    transp_test_final = []

    instLumi_test = []
    intLumiLHC_test = []
    infillLumi_test = []
    lastfillLumi_test = []
    filltime_test = []
    lastpointLumi_test = []
    true_time_test = []
    ring_index_test = []

    Norm_time_in_fill_test = []
    Norm_time_in_fill_prov_test = []
    weights_test = []
    FILL_num = []
    METADATA_test=[]
        
    for m_key, m_value in dictionary_test.items():

        data=np.load(f"{data_folder}/{m_value[0]}")
        data_df = pd.DataFrame(data)
        data_df.head()

        mean=[]
        for k in range (0, len(data_df.axes[1])):
            mean = np.append(mean, np.mean(data_df[k]))

        #mean transparency in iring selected for validation
        mean=mean[mean != -1]
        metadata = metadata.iloc[:len(mean)][mean != -1]

        #selecting metadata for validation fills 
        metadata_test = metadata[metadata.fill_num.isin(m_value[1])]
        metadata_test = metadata_test[(metadata_test.lumi_inst >= 0.0001*1e9) & (metadata_test.lumi_inst <= 0.0004*1e9) & (metadata_test.lumi_in_fill >= 0.1*1e9)]
        transp_test = mean[metadata_test.index.values[0]:metadata_test.index.values[0]+len(metadata_test.axes[0])]
        fill_num_test = metadata_test.fill_num.unique()

        FILL_num = np.append(FILL_num, fill_num_test)
        #METADATA_test = pd.concat(METADATA_test, metadata_test)
        METADATA_test.append(metadata_test) 

        #normalizing time_in_fill for test fills
        for k in fill_num_test:
            dftest = metadata_test[metadata_test.fill_num == k]
            Norm_time_in_fill_prov_test = dftest.loc[:,'time_in_fill']
            c=dftest['time_in_fill'].iloc[0]
            Norm_time_in_fill_prov_test = Norm_time_in_fill_prov_test/c
            Norm_time_in_fill_test=np.append(Norm_time_in_fill_test,Norm_time_in_fill_prov_test)

        for k in fill_num_test:
            df_test = metadata_test[metadata_test.fill_num == k]
            transp_test = [mean[i] for i in df_test.index.values]
            transp_test = transp_test/transp_test[0]
            transp_test_final = np.append(transp_test_final, transp_test)

        instLumi_test_p = (1e-9)*metadata_test.loc[:,'lumi_inst']
        intLumiLHC_test_p = (1e-9)*metadata_test.loc[:,'lumi_int']
        infillLumi_test_p = (1e-9)*metadata_test.loc[:,'lumi_in_fill']
        lastfillLumi_test_P = (1e-9)*metadata_test.loc[:,'lumi_last_fill']
        filltime_test_p = (1e-9)*metadata_test.loc[:,'time_in_fill']
        lastpointLumi_test_p = (1e-9)*metadata_test.loc[:, 'lumi_since_last_point']
        true_time_test_p = (1e-9)*metadata_test.loc[:, 'time']
        ring_index_test_p = np.zeros(len(metadata_test))
        for iev in range (0, len(metadata_test)):
            ring_index_test_p[iev] = m_key

        instLumi_test=np.append(instLumi_test, instLumi_test_p)
        intLumiLHC_test=np.append(intLumiLHC_test,intLumiLHC_test_p)
        infillLumi_test=np.append(infillLumi_test,infillLumi_test_p)
        lastfillLumi_test=np.append(lastfillLumi_test,lastfillLumi_test_P)
        filltime_test=np.append(filltime_test,filltime_test_p)
        lastpointLumi_test=np.append(lastpointLumi_test,lastpointLumi_test_p)
        true_time_test=np.append(true_time_test,true_time_test_p)
        ring_index_test=np.append(ring_index_test,ring_index_test_p)

        #validation weights for sample weighting 
    for k in range (0, len(Norm_time_in_fill_test)):
        if Norm_time_in_fill_test[k] == 1. :
            weight = 0.001/(len(FILL_num))
            weights_test.append(weight)
        else:
            #weight = (1-0.5)/ (len(fill_num2)+len(merged_norm_time_in_fill)-(len(fill_num)+len(fill_num1)))
            weight = (1-0.001)/(len(Norm_time_in_fill_test)-(len(FILL_num)))
            weights_test.append(weight)
        
        #merge everything into single objects
    all_inputs_test = np.stack((instLumi_test, intLumiLHC_test, infillLumi_test, lastfillLumi_test, filltime_test, lastfillLumi_test, ring_index_test), axis=-1)
    return [all_inputs_test, transp_test_final, weights_test, metadata_test]

#---------------------------------------------"SMOOTHING"-------------------------------------------------------
#--------Fill excluded from iring 22
nonsmooth22 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
                6030, 6041, 6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
                6152, 6159, 6160, 6167, 6168, 6192, 6193, 6263, 6318,

             #we exclude some fills for validation for iring 22
                 6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                 ]

#--------Fill excluded from iring 23
nonsmooth23 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
                6030, 6041, 6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
                6152, 6159, 6160, 6167, 6168, 6192, 6193, 6263, 6318,

             #we exclude some fills for validation for iring 23
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ]

#--------Fill excluded from iring 24 
nonsmooth24 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ]

#--------Fill excluded from iring 25
nonsmooth25 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ]            
    #with this command we actually exclude nonsmooth fills from the "fill" array 
    #but we are also excluding validation fills which we insert in the validation dictionary (dictionary_test).

#--------Fill excluded from iring 26
nonsmooth26 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ] 

#--------Fill excluded from iring 27
nonsmooth27 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ] 

#--------Fill excluded from iring 28
nonsmooth28 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
                ] 


#Fill used for inference
filltest24 = [5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371]

#the best thing to organize the space work might be a file json to control fills for validation 
#and train and npy files for transparency data

#running the functions below we have data ready for the model training
#mancano solo i pesi viruali per la loss 
dictionary_train = {22:["iRing22new.npy", nonsmooth22], 23:["iRing23new.npy", nonsmooth23], 24:["iRing24new.npy", nonsmooth24], 25:["iRing25new.npy", nonsmooth25], 26:["iRing26new.npy", nonsmooth26], 27:["iRing27new.npy", nonsmooth27], 28:["iRing28new.npy", nonsmooth28]}
dictionary_test = {24:["iRing24new.npy", filltest24]}
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

#Pre-processing data for training and validation
[all_inputs_train, transp_train, weights_train]=Pre_Processing_Train(dictionary_train, metadata)
[all_inputs_validation, transp_validation, weights_validation, metadata_test]=Pre_Processing_Test(dictionary_test,metadata)

#--------------------------------Machine Learning ─=≡Σ(([ ⊐•̀⌂•́]⊐-----------------------------------------------

def delta_train(transp_training,transp_predicted_train):
    loss = K.square(transp_training-transp_predicted_train)
    loss=loss*weights_train
    loss=K.sum(loss, axis=1)
    return loss

#test loss obtained with sample weighting
def delta_test(transp_test_final,transp_predicted_test):
    loss = K.square(transp_test_final-transp_predicted_test)
    loss=loss*weights_validation
    loss=K.sum(loss, axis=1)
    return loss

#DNN structure
inputs = Input(shape=(7,))
hidden1 = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(inputs)
drop1=Dropout(0.2)(hidden1)
hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(drop1)
drop2=Dropout(0.2)(hidden2)
hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(drop2)
outputs = Dense(1) (hidden3)

#model checkpoint and early stopping
filepath = "/home/federico/root/root-6.24.06-install/weights"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=0, save_best_only=True, mode='min')
callback_list=[checkpoint]

early_stopping = EarlyStopping(monitor = 'loss', patience=50, verbose=0, restore_best_weights= True)
model = Model ( inputs=inputs, outputs=outputs )

#model.add_loss( custom_loss( ,outputs, inputs) )

model.compile(loss = delta_train, optimizer='adam', metrics=[delta_train])

#write the summary of the network
model.summary()

#plot the network
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
)

all_inputs_training   = all_inputs_train
all_inputs_validation = all_inputs_validation
transp_training   = transp_train
transp_validation = transp_validation
#print(all_inputs_validation)
print('forma dellinput prima di model.fit')
#print(len(inputs(7)))
#print(model.layers[0,:])
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation), epochs=500, verbose=2, callbacks=[early_stopping])

#plot the training loss
plt.plot( history.history["loss"], label = 'train' )
plt.plot( history.history["val_loss"], label = 'validation' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.legend()
plt.show()


transp_predicted_validation = model.predict(all_inputs_validation)
transp_predicted_train = model.predict(all_inputs_train)

#prediction_single_fill=[]

#plot Transparency vs abs time of validation
plt.plot(metadata_test.time, transp_validation, ".b-", markersize=3, linewidth=0.75, label="measured")
plt.plot(metadata_test.time, transp_predicted_validation, ".r-", markersize=3, linewidth=0.75, label="predicted")
plt.xlabel("absolute time")
plt.ylabel("mean transparency")
plt.tick_params(labelsize=7)
plt.title(f"iRing ")
plt.legend()
plt.show()
plt.savefig("Predicitons_single_fill.png")
