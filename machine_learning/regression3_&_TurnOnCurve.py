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
#transparency data for a selected iRing#
data24=np.load(f"{data_folder}/iRing23new.npy")
data24_df = pd.DataFrame(data24)
data24_df.head()
mean24=[]


#compute mean transparnecy in iRings 
for i in range (0, len(data24_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean24 = np.append(mean24, np.mean(data24_df[i]))
mean24=mean24[mean24 != -1]
metadata = metadata.iloc[:len(mean24)][mean24 != -1]

#selectin metadata for fill (locking metadata to in_fill=1)
fill=metadata["fill_num"].unique()
fill = fill[fill != 0]
nonsmooth = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
             6030, 6041,
             6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
             6152, 6159, 6160, 6167, 6168,
             #6253, 6255, 
             6192, 6193, #6262,
             6263, 6318
             #6279 # non-smooth fills : 22/138
             #6349
#--------second tranche


             ]
#print(len(nonsmooth))

for iev in range (0, len(nonsmooth)) :
    #print(nonsmooth[iev])
    fill = fill[fill != nonsmooth[iev]]


#test fill
#escludo fill di test
fill=fill[fill != 6371]
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()
#devo togliere i fill non smooth dal dataset
print("fills usati per il train")
print(fill_num)

transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []
#riempie il vettore di transparency
for k in fill_num:
    #in metadata fill ci sono solo i dati relativi a ifferenti fills, 
    # quindi ha la stessa dimensione della transp_fill

    #restringe metadata a quello di un solo fill (k)
    #viene fatto per ogni fill6053
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
#a transp_fill corrispondono i dati matadata_fill
#me build the same:

#gets easyer data manipulation
#put metadata into arrays
instLumi = (1e-9)*metadata_fill.loc[:,'lumi_inst']
intLumiLHC = (1e-9)*metadata_fill.loc[:,'lumi_int']
infillLumi = (1e-9)*metadata_fill.loc[:,'lumi_in_fill']
lastfillLumi = (1e-9)*metadata_fill.loc[:,'lumi_last_fill']
filltime = (1e-9)*metadata_fill.loc[:,'time_in_fill']
lastpointLumi = (1e-9)*metadata_fill.loc[:, 'lumi_since_last_point']
true_time = (1e-9)*metadata_fill.loc[:, 'time']

all_inputs_train=np.stack((instLumi, infillLumi, intLumiLHC, lastfillLumi, filltime), axis=-1)
#validation dataset
filltest = 6371
metadata_6371 = metadata[metadata.fill_num == filltest]
metadata_6371 = metadata_6371[(metadata_6371.lumi_inst >= 0.0001*1e9) & (metadata_6371.lumi_inst <= 0.0004*1e9) & (metadata_6371.lumi_in_fill >= 0.1*1e9)]
transp_6371 = mean24[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
# normalizzo il dato di trasparenza a quella precedente al fill
transp_6371 = transp_6371/transp_6371[0]

instLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_inst']
intLumiLHC_test = (1e-9)*metadata_6371.loc[:,'lumi_int']
infillLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_in_fill']
lastfillLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_last_fill']
filltime_test = (1e-9)*metadata_6371.loc[:,'time_in_fill']
lastpointLumi_test = (1e-9)*metadata_6371.loc[:, 'lumi_since_last_point']
true_time_test = (1e-9)*metadata_6371.loc[:, 'time']


all_inputs_test=np.stack((instLumi_test, infillLumi_test, intLumiLHC_test, lastfillLumi_test, filltime_test), axis=-1)
#use weight decay and dropout together is usefull??
#see article on google docs 
# Machine learning  ─=≡Σ(([ ⊐•̀⌂•́]⊐
#DNN structure
inputs = Input(shape=(5,))

hidden1 = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(inputs)

drop1=Dropout(0.3)(hidden1)

hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(drop1)

drop2=Dropout(0.2)(hidden2)

hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(drop2)

#drop3=Dropout(0.20)(hidden3)

#hidden4 = Dense(16, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(hidden3)

#drop3=Dropout(0.20)(hidden4)

#hidden5 = Dense(32, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-9, l2=1e-10), bias_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l2(1e-10))(hidden4)

#hidden6 = Dense(32, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-10, l2=1e-13), bias_regularizer=regularizers.l2(1e-13), activity_regularizer=regularizers.l2(1e-13))(hidden5)

outputs = Dense(1) (hidden3)#kernel_regularizer=regularizers.l1_l2(l1=1e-16, l2=1e-16), bias_regularizer=regularizers.l2(1e-16), activity_regularizer=regularizers.l2(1e-16))(hidden3)

#model checkpoint and early stopping
filepath = "/home/federico/root/root-6.24.06-install/weights"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
callback_list=[checkpoint]
early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=150, verbose=1, restore_best_weights= True)

model = Model ( inputs=inputs, outputs=outputs )
model.compile(loss='MSE', optimizer='adam', metrics=['mean_squared_error'])

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
transp_training   = transp_fill
transp_validation = transp_6371

#print(all_inputs_validation)

#now actually performing the train (ง •̀_•́)ง
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation), epochs=500, verbose=2, callbacks=[early_stopping])

#plot the training loss
plt.plot( history.history["loss"], label = 'train' )
plt.plot( history.history["val_loss"], label = 'validation' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.legend()
plt.show()

# plot MSE
plt.plot(history.history["mean_squared_error"], label = 'train')
plt.plot(history.history["val_mean_squared_error"], label='validation')
plt.legend()
plt.show()

#now test the performance of the DNN ಠ_ರೃ
transp_predicted_validation = model.predict(all_inputs_validation)
#plot on abs time of metadata for a selected fill_num
plt.plot(metadata_6371.time, transp_validation, ".b-", markersize=3, linewidth=0.75, label="measured")
plt.plot(metadata_6371.time, transp_predicted_validation, ".r-", markersize=3, linewidth=0.75, label="predicted")
plt.xlabel("time")
plt.ylabel("mean transparency")
plt.tick_params(labelsize=7)
plt.title(f"predicted & observed - fill{filltest}")
plt.show()
print("lunghezza della t ")
print(len(transp_predicted_validation))
#plot integrated lumi vs transparency 
plt.plot(metadata_6371.lumi_int, transp_validation, ".m-", markersize=3, linewidth=0.75, label="measured")
plt.plot(metadata_6371.lumi_int, transp_predicted_validation, ".c-", markersize=3, linewidth=0.75, label="predicted")
plt.xlabel("time")
plt.ylabel("mean transparency")
plt.tick_params(labelsize=7)
plt.title(f"predicted & observed - fill{filltest}")
plt.show()
print("lunghezza della t ")
print((transp_predicted_validation))

error = 0
#final mse 
for i in range (0, len(transp_validation)):
    error = error + (transp_validation[i]-transp_predicted_validation[i])*(transp_validation[i]-transp_predicted_validation[i])

#mean square error: 1/N sum((O-P)^2)
mse = error/len(transp_validation)
print(f"mean square error for fill {filltest}")
print(mse)

#--------Turn_On_Curve---------#
nbin = 400
minimo = 0
massimo = 60
threshold = 30
delta_value = (massimo-minimo)/nbin
nEvents = 1000

print("metti input num_fill")
fill = input()
#input validation fill_num

selected_metadata = metadata[metadata.fill_num == int(fill)]
selected_transp = [mean24[i] for i in selected_metadata.index.tolist()]
#prepare

lumi_in_fill = selected_metadata.lumi_in_fill.to_numpy()
lumi_inst = selected_metadata.lumi_inst.to_numpy()
lumi_inst_0 = np.empty(np.size(selected_transp))
lumi_inst_0.fill(lumi_inst[0])
#-------Without correction---------#
c0 = ROOT.TCanvas("cc_turn_on", "", 800, 700)

hist0 = ROOT.TH1F("without correction", "", nbin, minimo, massimo)

for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(selected_transp)):
            

            value_smeared = value*selected_transp[i]
            
            if value_smeared > threshold:
                
                hist0.Fill(value) #pesare per luminosità istantanea


hist0.Scale(1./(nEvents*np.size(selected_transp)))
hist0.SetLineWidth(2)
hist0.SetLineColor(600)
hist0.SetFillColor(600)

hist0.Draw()
hist0.GetXaxis().SetTitle("Energy [GeV]")
hist0.GetYaxis().SetTitle("Efficiency")
#c0.Draw()
#vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
#vertical_line.Draw()
c0.cd
#c0.SaveAs("h_turn_on.png")

#-------------With correction--------------#

#c0 = ROOT.TCanvas("cc_turn_on", "", 800, 700)
hist = ROOT.TH1F("with correction", "", nbin, minimo, massimo)
transparency = transp_predicted_validation

for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(transparency)):
            value_smeared = value*transparency[i]
            if value_smeared > threshold:
                hist.Fill(value)


hist.Scale(1./(nEvents*np.size(selected_transp)))
hist.SetLineWidth(2)
hist.SetLineColor(632)
hist.SetFillColor(632)
hist.Draw("SAME")
hist.GetXaxis().SetTitle("Energy [GeV]")
hist.GetYaxis().SetTitle("Efficiency")
#c0.Draw("same")
vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
vertical_line.Draw()


# mg=ROOT.THStack("hs","Stacked 1D histograms")
# mg.Add(hist0)
# mg.Add(hist)
# mg.Draw("LP")
c0.Draw()
c0.SaveAs("TurnOnCurve.png")
#a parte la turn on curve va tutto bene, ora devo inserire dati di altri i Ring 