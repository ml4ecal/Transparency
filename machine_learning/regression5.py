from cmath import exp
from itertools import tee
from tkinter.ttk import LabelFrame
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
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import get_session


#------

#Libraries for machine learning
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

import seaborn as sns

data_folder = ('/home/federico/root/root-6.24.06-install')
#data_folder = ('/gwpool/users/fdematteis')

metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

#----------------------------------------------------------
#transparency data for a selected iRing : the target function

#23
data24=np.load(f"{data_folder}/iRing23new.npy")
data24_df = pd.DataFrame(data24)
data24_df.head()
mean24=[]
#25
data25=np.load(f"{data_folder}/iRing25new.npy")
data25_df = pd.DataFrame(data25)
data25_df.head()
mean25=[]
#26
data26=np.load(f"{data_folder}/iRing26new.npy")
data26_df = pd.DataFrame(data26)
data26_df.head()
mean26=[]
#compute mean transparnecy in iRings 
for i in range (0, len(data24_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean24 = np.append(mean24, np.mean(data24_df[i]))

for i in range (0, len(data25_df.axes[1])):
    mean25 = np.append(mean25, np.mean(data25_df[i]))

for i in range (0, len(data26_df.axes[1])):
    mean26 = np.append(mean26, np.mean(data26_df[i]))


#iring23

mean24=mean24[mean24 != -1]
metadata = metadata.iloc[:len(mean24)][mean24 != -1]
#iring25
mean25=mean25[mean25 != -1]
metadata1 = metadata.iloc[:len(mean25)][mean25 != -1]
#iring26
mean26=mean26[mean26 != -1]
metadata2 = metadata.iloc[:len(mean26)][mean26 != -1]

#selecting metadata for fill (locking metadata to in_fill=1)
fill=metadata["fill_num"].unique()
fill = fill[fill != 0]

fill1=metadata1["fill_num"].unique()
fill1 = fill1[fill1 != 0]

fill2=metadata2["fill_num"].unique()
fill2 = fill2[fill2 != 0]

nonsmooth = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
             6030, 6041,
             6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
             6152, 6159, 6160, 6167, 6168,
             6192, 6193, 
             6263, 6318,
             #escludo i fill che userò per il train
             6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
             
#--------second tranche
             ]
nonsmooth1 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
              5984,  6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 6159,
              6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 6300, 6318, 6348,
              6349,
              #escludo i fill che userò per il train
              6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046
              ]
             
for iev in range (0, len(nonsmooth)) :
    #print(nonsmooth[iev])
    fill = fill[fill != nonsmooth[iev]]

for iev in range (0, len(nonsmooth1)) :
    #print(nonsmooth[iev])
    fill1 = fill1[fill1 != nonsmooth1[iev]]


#sbloccare quando inserisco i fill da escludere (sopra)
# for iev in range (0, len(nonsmooth1)) :
#     #print(nonsmooth[iev])
#     fill2 = fill2[fill2 != nonsmooth2[iev]]


#ora escludo i fill che ho deselezionato dai dati di train
#i fill di validation sono gia stati esclusi
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

metadata_fill1 = metadata[metadata.fill_num.isin(fill1)]
metadata_fill1 = metadata_fill1[(metadata_fill1.lumi_inst >= 0.0001*1e9) & (metadata_fill1.lumi_inst <= 0.0004*1e9) & (metadata_fill1.lumi_in_fill >= 0.1*1e9)]
fill_num1 = metadata_fill1.fill_num.unique()
#fatto!


transp_fill = []
transp_fill1 = []
transp_fill2 = []
lumi_inst_0 = []
lumi_int_0 = []


#calcolo la main transparency per ognuno dei due iRing
for k in fill_num:
#transparency relativa ai fill selezionati
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean24[i] for i in df.index.values]
    #transp ha la grandezza del dataframe ristretto al k esimo fill
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)
    #in transp_fill ci sono i dati di trasparenza normalizzata per ogni fill;
#-------
for k in fill_num1:
    df1 = metadata_fill1[metadata_fill1.fill_num == k]
    transp1 = [mean25[i] for i in df1.index.values]
    transp1 = transp1/transp1[0]
    transp_fill1 = np.append(transp_fill1, transp1)


#-----iring23 
instLumi = (1e-9)*metadata_fill.loc[:,'lumi_inst']
intLumiLHC = (1e-9)*metadata_fill.loc[:,'lumi_int']
infillLumi = (1e-9)*metadata_fill.loc[:,'lumi_in_fill']
lastfillLumi = (1e-9)*metadata_fill.loc[:,'lumi_last_fill']
filltime = (1e-9)*metadata_fill.loc[:,'time_in_fill']
lastpointLumi = (1e-9)*metadata_fill.loc[:, 'lumi_since_last_point']
true_time = (1e-9)*metadata_fill.loc[:, 'time']

ring_index = np.zeros(len(metadata_fill))
for i in range (0, len(ring_index)):
    ring_index[i] = 23

#-----iring25
instLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_inst']
intLumiLHC1 = (1e-9)*metadata_fill1.loc[:,'lumi_int']
infillLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_in_fill']
lastfillLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_last_fill']
filltime1 = (1e-9)*metadata_fill1.loc[:,'time_in_fill']
lastpointLumi1 = (1e-9)*metadata_fill1.loc[:, 'lumi_since_last_point']
true_time1 = (1e-9)*metadata_fill1.loc[:, 'time']

ring_index1 = np.zeros(len(metadata_fill1))
for i in range (0, len(ring_index1)):
    ring_index1[i] = 25

#now i have to merge transparency datas and luminosity metadatas
#metadata merge
merged_instLumi = np.append(instLumi, instLumi1)
merged_intLumiLHC = np.append(intLumiLHC, intLumiLHC1)   
merged_infillLumi = np.append(infillLumi, infillLumi1)
merged_lastfillLumi = np.append(lastfillLumi, lastfillLumi1)
merged_filltime = np.append(filltime, filltime1)
merged_ring_index = np.append(ring_index, ring_index1)
#transparency merge 

#transp_train = np.append(transp_fill, transp_fill1)
transp_train=transp_fill
#all_inputs_train=np.stack((instLumi, infillLumi, intLumiLHC, filltime, ring_index, lastfillLumi), axis=-1)

#Validation dataset
#fill usati per il test
#filltest = [6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046]
filltest = [5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371]
metadata_test = metadata[metadata.fill_num.isin(filltest)]

metadata_test = metadata_test[(metadata_test.lumi_inst >= 0.0001*1e9) & (metadata_test.lumi_inst <= 0.0004*1e9) & (metadata_test.lumi_in_fill >= 0.1*1e9)]
#estraggo transparency per il test
transp_test = mean24[metadata_test.index.values[0]:metadata_test.index.values[0]+len(metadata_test.axes[0])]
fill_num_test = metadata_test.fill_num.unique()

#normalizzo i dati di trasparenza per il test
#è sbagliato normalizzare così, devo normalizzare nello stesso modo usato per i metadati di train
transp_test_final=[]

for k in fill_num_test:
    df_test = metadata_test[metadata_test.fill_num == k]
    #sto scegliendo i dati di trasparenza dall'iRing 23 che quì è indicato con 24
    transp_test = [mean24[i] for i in df_test.index.values]
    transp_test = transp_test/transp_test[0]
    transp_test_final = np.append(transp_test_final, transp_test)

print(transp_test_final)

#in metadata_test ci sono i metadati relativi ai fill che uso per il train

#Ora devo preparare i metadati di validation usando metadata_test

instLumi_test = (1e-9)*metadata_test.loc[:,'lumi_inst']
intLumiLHC_test = (1e-9)*metadata_test.loc[:,'lumi_int']
infillLumi_test = (1e-9)*metadata_test.loc[:,'lumi_in_fill']
lastfillLumi_test = (1e-9)*metadata_test.loc[:,'lumi_last_fill']
filltime_test = (1e-9)*metadata_test.loc[:,'time_in_fill']
lastpointLumi_test = (1e-9)*metadata_test.loc[:, 'lumi_since_last_point']
true_time_test = (1e-9)*metadata_test.loc[:, 'time']



ring_index_test = np.zeros(len(metadata_test))
for iev in range (0, len(metadata_test)):
    ring_index_test[iev] = 23

#all_inputs_test=np.stack((instLumi_test, infillLumi_test, intLumiLHC_test, filltime_test, ring_index_test, lastfillLumi_test, Norm_time_in_fill_test), axis=-1)

# Machine learning  ─=≡Σ(([ ⊐•̀⌂•́]⊐
#defining a custom loss function
import keras.backend as K

#validation time_in_fill normalizzato
Norm_time_in_fill_test=[]
Norm_time_in_fill_prov_test=[]
exp_time_in_fill_prov=[]
weightstest=[]
for k in fill_num_test:
    dftest = metadata_test[metadata_test.fill_num == k]
    
    #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
    Norm_time_in_fill_prov_test = dftest.loc[:,'time_in_fill']
    c=dftest['time_in_fill'].iloc[0]
    Norm_time_in_fill_prov_test = Norm_time_in_fill_prov_test/c
    Norm_time_in_fill_test=np.append(Norm_time_in_fill_test,Norm_time_in_fill_prov_test)
    weightstest.append((0.6/len(fill_num_test)))
    for i in range (1, len(dftest)):
        #weights.append(0)
        weightstest.append((1-0.6)/(len(transp_train)-len(fill_num_test)))
    #print('time in fill normalizzato')
    #print(Norm_time_in_fill_test)

all_inputs_test=np.stack((instLumi_test, infillLumi_test, intLumiLHC_test, ring_index_test, lastfillLumi_test, Norm_time_in_fill_test), axis=-1)


#TRAIN
Norm_time_in_fill_train=[]
Norm_time_in_fill_prov_train=[]
weightstrain=[]

for k in fill_num:
    dftrain = metadata_fill[metadata_fill.fill_num == k]
    
    #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
    Norm_time_in_fill_prov_train = dftrain.loc[:,'time_in_fill']
    Norm_time_in_fill_prov_train = Norm_time_in_fill_prov_train/dftrain['time_in_fill'].iloc[0]
    Norm_time_in_fill_train=np.append(Norm_time_in_fill_train,Norm_time_in_fill_prov_train)
    #weights.append(1)
    weightstrain.append((0.01/len(fill_num)))
    for i in range (1, len(dftrain)):
        #weights.append(0)
        weightstrain.append((1-0.01)/(len(transp_train)-len(fill_num)))
    #print('time in fill normalizzato')
    #print(Norm_time_in_fill_train)
    #print(len(Norm_time_in_fill))



all_inputs_train=np.stack((instLumi, infillLumi, intLumiLHC,  ring_index, lastfillLumi, Norm_time_in_fill_train), axis=-1)
print(len(all_inputs_test))


#print('tempo in fill norm test')
#print(len(Norm_time_in_fill_test))
#print('tempo in fill norm train')
#print(len(Norm_time_in_fill_train))
#provo a definire un vettore con dentro gli esponenziali:
    # for k in range (0,len(df)):
    #     exp_time_in_fill_prov = exp(Norm_time_in_fill_prov_test[k] -c)
    #     print(exp_time_in_fill_prov)

    # exp_time_in_fill = np.append(exp_time_in_fill, exp_time_in_fill_prov)

from math import e
proof=e**0
print('esponenziale di 0')
print(proof)
plt.plot(Norm_time_in_fill_test, (1+10*e**(-1000*(Norm_time_in_fill_test-1))), ".b", markersize=3, linewidth=0.75)
plt.xlabel("Time_in_fill")
plt.ylabel("exp(-(time_in_fill-1))")
plt.tick_params(labelsize=7)
plt.title('fattore moltiplicativo della loss function')
plt.legend()
plt.show()
plt.show()
custom_delta=[]

#generic version
# def custom_loss(y_true,y_pred,inputs):
#     def loss (y_true,y_pred):       
#         return K.square(y_true-y_pred) * 10*e**(-1000*(inputs(7,)-1))
#     return loss

#train oriented version
# def delta(transp_training,transp_predicted_train):
#     return ((transp_training-transp_predicted_train)**2)/tf.cast(len(transp_predicted_train), tf.float32) * (1+ 10*e**(-1000*(Norm_time_in_fill_train-1)))

#train oriented with smple weighting
def delta_train(transp_training,transp_predicted_train):
    loss = K.square(transp_training-transp_predicted_train)
    loss=loss*weightstrain
    loss=K.sum(loss, axis=1)
    return loss

#test loss obtained with sample weighting
def delta_test(transp_test_final,transp_predicted_test):
    loss = K.square(transp_test_final-transp_predicted_test)
    loss=loss*weightstest
    loss=K.sum(loss, axis=1)
    return loss


def custom_loss (inputs):
    def delta (y_true,y_pred):
        loss = K.square(y_true-y_pred) * 10*e**(-1000(inputs(7,)-1))
        loss=K.sum(loss, axis=1)
        return loss
    return delta


def delta_mse (transp_training,transp_predicted_train):
    delta  = K.square(transp_training-transp_predicted_train) * 10*e**(-1000(Norm_time_in_fill_train-1))
    delta = K.sum(delta, axis=1)
    return delta

#DNN structure
inputs = Input(shape=(6,))
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
early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=0, restore_best_weights= True)


model = Model ( inputs=inputs, outputs=outputs )

#model.add_loss( custom_loss( ,outputs, inputs) )

model.compile(loss = delta_train, optimizer='adam', metrics=delta_train)

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
all_inputs_validation = all_inputs_test
transp_training   = transp_train
transp_validation = transp_test_final
#print(all_inputs_validation)
print('forma dellinput prima di model.fit')
#print(len(inputs(7)))
#print(model.layers[0,:])
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_test_final), epochs=300, verbose=2, callbacks=[early_stopping])

#plot the training loss
plt.plot( history.history["loss"], label = 'train' )
plt.plot( history.history["val_loss"], label = 'validation' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.legend()
plt.show()

# # plot MSE
# plt.plot(history.history["mean_squared_error"], label = 'train')
# plt.plot(history.history["val_mean_squared_error"], label='validation')
# plt.legend()
# plt.show()


transp_predicted_validation = model.predict(all_inputs_validation)
transp_predicted_train = model.predict(all_inputs_train)

prediction_single_fill=[]
#plot on abs time of metadata for a selected fill_num


plt.plot(metadata_test.time, transp_validation, ".b-", markersize=3, linewidth=0.75, label="measured")
plt.plot(metadata_test.time, transp_predicted_validation, ".r-", markersize=3, linewidth=0.75, label="predicted")
plt.xlabel("absolute time")
plt.ylabel("mean transparency")
plt.tick_params(labelsize=7)
plt.title(f"fill {fill_num_test}")
plt.legend()
plt.show()
plt.savefig("Plot generated using Matplotlib.png")

# m=0
# for k in fill_num_test:
#     df_test = metadata_test[metadata_test.fill_num == k]
#     transp_test = [mean24[i] for i in df_test.index.values]
#     transp_test = transp_test/transp_test[0]
#     print('lunghezza del primo fill')
#     print(len(transp_test))
#     for j in range (0, len(transp_test)):
#         prediction_single_fill[j] = transp_predicted_validation[m+j] 

#     m=m+j+1
#         #alla fine del primo for m è lungo quanto il primo fill: len(transp_test)
#         #nel secondo for m parte da len(transp_test)+1 e va fino alla lunghezza del secondo fill
#         #sempre in questo for posso calcolare mse e salvarlo in un array
#     #k=len(transp_test)+1
#     plt.plot(metadata_test.time, transp_test, ".b-", markersize=3, linewidth=0.75, label="measured")
#     plt.plot(metadata_test.time, prediction_single_fill, ".r-", markersize=3, linewidth=0.75, label="predicted")
#     plt.xlabel("absolute time")
#     plt.ylabel("mean transparency")
#     plt.tick_params(labelsize=7)
#     plt.title(f"fill {fill_num_test}")
#     plt.legend()
#     plt.show()


#calcolare mse per ogni fill
#usare time_in_fill
#mean square error: 1/N sum((O-P)^2)
mse = ((transp_validation-transp_predicted_validation)**2).mean(axis=0)
print(f"mean square error for fill {filltest}")
print(mse)

# # #-----TurnOnCurve------
# nbin = 400
# minimo = 0
# massimo = 60
# threshold = 30
# delta_value = (massimo-minimo)/nbin
# nEvents = 1000

# #print("metti input num_fill")
# fill = filltest#input()
# #input validation fill_num

# selected_metadata = metadata[metadata.fill_num == int(fill)]
# selected_transp = [mean24[i] for i in selected_metadata.index.tolist()]
# #prepare

# lumi_in_fill = selected_metadata.lumi_in_fill.to_numpy()
# lumi_inst = selected_metadata.lumi_inst.to_numpy()
# lumi_inst_0 = np.empty(np.size(selected_transp))
# lumi_inst_0.fill(lumi_inst[0])


# canvas0 = ROOT.TCanvas("TurnOn without correction", "", 800, 700)
# #-----step function-----
# hist1 = ROOT.TH1F("ideal-step function", "", nbin, minimo, massimo)
# transparency = transp_predicted_validation
# for ibin in range(0, nbin):
#     value = minimo+(ibin+0.2)*delta_value
#     for iEvent in range(0, nEvents):
#         for i in range(0, np.size(transparency)):
#             value_smeared = value
#             if value_smeared > threshold:
#                 hist1.Fill(value)
#                 #hist.Eval(  , "A")

# hist1.Scale(1./(nEvents*np.size(transp_predicted_validation)))
# hist1.SetLineWidth(1)
# hist1.SetLineColor(1)
# hist1.Draw("h ")
# hist1.SetTitle(f"trigger efficiency - fill {filltest}")
# hist1.GetXaxis().SetTitle("Energy [GeV]")
# hist1.GetYaxis().SetTitle("Efficiency")
# hist1.SetStats(000000000)


# # #----TurnOn without correction
# hist0 = ROOT.TH1F("without correction", "", nbin, minimo, massimo)
# for ibin in range(0, nbin):
#     value = minimo+(ibin+0.2)*delta_value
#     for iEvent in range(0, nEvents):
#         for i in range(0, np.size(selected_transp)):
#             value_smeared = value*selected_transp[i]
#             if value_smeared > threshold:
#                 hist0.Fill(value) #pesare per luminosità istantanea


# hist0.Scale(1./(nEvents*np.size(selected_transp)))
# hist0.SetLineWidth(1)
# hist0.SetLineColor(600)
# hist0.Draw("h same")
# hist0.GetXaxis().SetTitle("Energy [GeV]")
# hist0.GetYaxis().SetTitle("Efficiency")
# hist0.SetStats(000000000)
# canvas0.cd

# # #-------------With correction--------------#

# canvas0 = ROOT.TCanvas("Corrected TurnOn", "", 800, 700)
# hist = ROOT.TH1F("with correction", "", nbin, minimo, massimo)
# transparency = transp_predicted_validation

# for ibin in range(0, nbin):
#     value = minimo+(ibin+0.2)*delta_value
#     for iEvent in range(0, nEvents):
#         for i in range(0, np.size(transparency)):
#             value_smeared = value*transparency[i]
#             if value_smeared > threshold:
#                 hist.Fill(value)
#                 #hist.Eval(  , "A")


# hist.Scale(1./(nEvents*np.size(transp_predicted_validation)))
# hist.SetLineWidth(1)
# hist.SetLineColor(632)
# hist.Draw("h same")
# hist.SetTitle("trigger efficiency")
# hist.GetXaxis().SetTitle("Energy [GeV]")
# hist.GetYaxis().SetTitle("Efficiency")
# hist.SetStats(000000000)


# # legend = ROOT.TLegend(0.1,0.7,0.48,0.9)
# # legend.SetHeader("","C") #  option "C" allows to center the header
# # legend.AddEntry(hist,"With correction")
# # legend.AddEntry(hist0,"Without correction")
# # legend.AddEntry(hist1,"Ideal")
# # legend.Draw()

# # #-----Draw canvas
# canvas0.Draw()
# #vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
# #vertical_line.Draw()
# canvas0.SaveAs("TurnOnCurve.png")

# #---------------------------------------
# # mg=ROOT.THStack("hs","Stacked 1D histograms")
# # mg.Add(hist0)
# # mg.Add(hist)
# # mg.Draw("LP")
# #a parte la turn on curve va tutto bene, ora devo inserire dati di altri i Ring 


