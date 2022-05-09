from scipy import special
from itertools import tee
import matplotlib.pyplot as plt
import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np
#libraries for data analysis
from pprint import pprint
from collections import namedtuple
import datetime
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error
#Libraries for machine learning
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import regularizers

data_folder = ('/home/federico/root/root-6.24.06-install')

data23 = np.load(f"{data_folder}/iRing23new.npy")   
data23_df = pd.DataFrame(data23)
data23_df.head()
#mean transaprency in iRing
mean23=[]
for i in range (0, len(data23_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean23 = np.append(mean23, np.mean(data23_df[i]))
mean23 = mean23[mean23 != -1]

#metadata inputs from LHC
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
metadata = metadata.iloc[:len(mean23)][mean23!=-1]

#lock mean transparency in a selected fill for the validation dataset
#print("insert fill_num")
test_fill = 6371
selected_metadata_test = metadata[metadata.fill_num == int(test_fill)]
selected_transp_test = [mean23[i] for i in selected_metadata_test.index.values]
date_test = [datetime.datetime.fromtimestamp(ts) for ts in selected_metadata_test.time]

#ora dovrei avere selezionata la trasparenza relativa solo a metadata_fill

#train metadata transparency
#inizio con un solo train-set
#poi ne inserisco altri 5962 6297
train_fill1 = 5697
selected_metadata_train = metadata[metadata.fill_num == int(train_fill1)]
selected_transp_train = [mean23[i] for i in selected_metadata_train.index.values]
date_train = [datetime.datetime.fromtimestamp(ts) for ts in selected_metadata_train.time]

train_fill2 = 6291
selected_metadata_train2 = metadata[metadata.fill_num == int(train_fill2)]
selected_transp_train2 = [mean23[i] for i in selected_metadata_train2.index.values]
date_train = [datetime.datetime.fromtimestamp(ts) for ts in selected_metadata_train.time]

train_fill3 = 6297
selected_metadata_train3 = metadata[metadata.fill_num == int(train_fill3)]
selected_transp_train3 = [mean23[i] for i in selected_metadata_train3.index.values]
date_train = [datetime.datetime.fromtimestamp(ts) for ts in selected_metadata_train3.time]

#train_metadata = np.append(selected_metadata_train,selected_metadata_train2)
#train_metadata = np.append(train_metadata, selected_metadata_train3)

#selected_metadata_test = selected_metadata_test.to_numpy()
#selected_metadata_train=selected_metadata_train.to_numpy()
#le rows di metadata relative al fill 6371 sono 145
#la len di transp relativa al fill è 145
#posso ricavare fill num, in questo modo ho accesso145
#posso ricavare fill num, in questo modo ho acce
#a tutti i fill che voglio nel dataset e a tutte le
#transp relative ai fill.
#L'idea è scegliere alcuni fill, non tutti
#e in particolare vogliamo quelli più smooth.
#ora devo prendere i metadati
#--------------------------------------

transp_norm_test=[]
#transp = [selected_transp_test[i] for i in selected_transp_test.index.values]
transp = selected_transp_test/selected_transp_test[0]
transp_norm_test = np.append(transp_norm_test, transp)


transp_norm_train=[]
#transp = [selected_transp_train[i] for i in selected_transp_train.index.values]
transp = selected_transp_train/selected_transp_train[0]
transp_norm_train = np.append(transp_norm_train, transp)
print("trasparenza train 1")
print(transp_norm_train)

transp_norm_train2=[]
#transp = [selected_transp_train[i] for i in selected_transp_train.index.values]
transp = selected_transp_train2/selected_transp_train2[0]
transp_norm_train2 = np.append(transp_norm_train2, transp)

transp_norm_train3=[]
#transp = [selected_transp_train[i] for i in selected_transp_train.index.values]
transp = selected_transp_train3/selected_transp_train3[0]
transp_norm_train3 = np.append(transp_norm_train3, transp)

transp_norm_TRAIN=np.append(transp_norm_train, transp_norm_train2)
transp_norm_TRAIN2=np.append(transp_norm_TRAIN, transp_norm_train3)

#ho fatto l'append delle transp
print(len(selected_transp_test))
print(len(transp_norm_test))
print(len(selected_transp_train))
print(len(transp_norm_TRAIN))

#extract metadata inputs for validation
instLumi_test = (1e-9)*selected_metadata_test.loc[:,'lumi_inst']
intLumiLHC_test = (1e-9)*selected_metadata_test.loc[:,'lumi_int']
infillLumi_test = (1e-9)*selected_metadata_test.loc[:,'lumi_in_fill']
lastfillLumi_test = (1e-9)*selected_metadata_test.loc[:,'lumi_last_fill']
filltime_test = (1e-9)*selected_metadata_test.loc[:,'time_in_fill']
#lastpointLumi_test = (1e-9)*metadata_test.loc[:, 'lumi_since_last_point']
#true_time_test = (1e-9)*metadata_test.loc[:, 'time']

#extract metadata inputs for train
instLumi_train = (1e-9)*selected_metadata_train.loc[:,'lumi_inst']
intLumiLHC_train = (1e-9)*selected_metadata_train.loc[:,'lumi_int']
infillLumi_train = (1e-9)*selected_metadata_train.loc[:,'lumi_in_fill']
lastfillLumi_train = (1e-9)*selected_metadata_train.loc[:,'lumi_last_fill']
filltime_train = (1e-9)*selected_metadata_train.loc[:,'time_in_fill']
#lastpointLumi = (1e-9)*metadata_train.loc[:, 'lumi_since_last_point']
#true_time = (1e-9)*metadata_train.loc[:, 'time']

#extract metadata inputs for train
instLumi_train2 = (1e-9)*selected_metadata_train2.loc[:,'lumi_inst']
intLumiLHC_train2 = (1e-9)*selected_metadata_train2.loc[:,'lumi_int']
infillLumi_train2 = (1e-9)*selected_metadata_train2.loc[:,'lumi_in_fill']
lastfillLumi_train2 = (1e-9)*selected_metadata_train2.loc[:,'lumi_last_fill']
filltime_train2 = (1e-9)*selected_metadata_train2.loc[:,'time_in_fill']
#lastpointLumi = (1e-9)*metadata_train.loc[:, 'lumi_since_last_point']
#true_time = (1e-9)*metadata_train.loc[:, 'time']

#extract metadata inputs for train
instLumi_train3 = (1e-9)*selected_metadata_train3.loc[:,'lumi_inst']
intLumiLHC_train3 = (1e-9)*selected_metadata_train3.loc[:,'lumi_int']
infillLumi_train3 = (1e-9)*selected_metadata_train3.loc[:,'lumi_in_fill']
lastfillLumi_train3 = (1e-9)*selected_metadata_train3.loc[:,'lumi_last_fill']
filltime_train3 = (1e-9)*selected_metadata_train3.loc[:,'time_in_fill']
#lastpointLumi = (1e-9)*metadata_train.loc[:, 'lumi_since_last_point']
#true_time = (1e-9)*metadata_train.loc[:, 'time']

#append valori di trasparenza:
print("input shape")
instLumi_train=np.append(instLumi_train, instLumi_train2)
instLumi_train2=np.append(instLumi_train, instLumi_train3)
print(len(instLumi_train))

intLumiLHC_train = np.append(intLumiLHC_train, intLumiLHC_train2)
intLumiLHC_train2 = np.append(intLumiLHC_train, intLumiLHC_train3)
print(len(intLumiLHC_train))

infillLumi_train = np.append(infillLumi_train, infillLumi_train2)
infillLumi_train2 = np.append(infillLumi_train, infillLumi_train3)
print(len(infillLumi_train))

lastfillLumi_train = np.append(lastfillLumi_train, lastfillLumi_train2)
lastfillLumi_train2 = np.append(lastfillLumi_train, lastfillLumi_train3)
print(len(lastfillLumi_train))

filltime_train = np.append(filltime_train, filltime_train2)
filltime_train2 = np.append(filltime_train, filltime_train3)
print(len(filltime_train))

#add all inputs into one object
all_inputs_test=np.stack((instLumi_test, infillLumi_test, intLumiLHC_test, filltime_test, lastfillLumi_test), axis=-1)
all_inputs_train=np.stack((instLumi_train2, infillLumi_train2, intLumiLHC_train2, filltime_train2, lastfillLumi_train2), axis=-1)


plt.figure()
plt.scatter(all_inputs_test[:,0], all_inputs_test[:, 1], c=transp_norm_test, cmap=plt.cm.RdBu, edgecolors='k')
plt.show()
plt.figure()

## Machine Learning 

inputs = Input(shape=(5,))
#first layer con 500 neuroni, and f=relu
hidden1 = Dense(500, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-16), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(inputs)
#hidden = LSTM(32)(inputs)
#second layer with 100 neurons and f=relu

hidden2 = Dense(500, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-16), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(hidden1)

hidden3 = Dense(500, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-16), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(hidden2)

#dense = Dense(3, kernel_regularizer='l2')

outputs = Dense(1)(hidden3)
model = Model ( inputs=inputs, outputs=outputs )
#model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.compile(loss='MSE', optimizer='adam')
model.summary()

# plot the network
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
)

all_inputs_training   = all_inputs_train#dovrei sostituire con l'input metadata di train
all_inputs_validation = all_inputs_test
transp_training   = transp_norm_TRAIN2#per ora è quella del fill che metto in input
transp_validation = transp_norm_test#la riempio con un altro tarain

# now actually performing the train (ง •̀_•́)ง
#da errore sul train perchè vede 54mila vs 
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation), epochs=300, verbose=0)

# ... and plot the training loss
plt.plot( history.history["val_loss"] )
plt.plot( history.history["loss"] )
plt.show()

#infine devo riempire la TurnOn per il singolo fill usando TurnOnCurve.py



# now test the performance of the DNN ಠ_ರೃ

transp_predicted_validation = model.predict(all_inputs_validation)

plt.plot(selected_metadata_test.time, transp_validation, "b .")
plt.plot(selected_metadata_test.time, transp_predicted_validation, "r +")
plt.show()
#print("Predicted mean Transparency in iRing")
#print(transp_predicted_validation)
