#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ROOT
import numpy as np
import matplotlib.pyplot as pltt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special
import random


# In[ ]:


def transp_func(data,a,b,c,d,e,f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))


# In[ ]:


def transp_func_2(data,a,b,c,d,e,f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*y)+(1-d)*np.exp(f*y))/(d*np.exp(-e*y_0)+(1-d)*np.exp(f*y_0))


# In[ ]:


data_folder=("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/DataPreparation")


# In[ ]:


#Read metadata
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")


# In[ ]:


#Load data
data = np.load(f"{data_folder}/iring_25.npy")
data_test = np.load(f"{data_folder}/iring_26.npy")
data_test_2 = np.load(f"{data_folder}/iring_24.npy")

data_df = pd.DataFrame(data)
data_df_test = pd.DataFrame(data_test)
data_df_test_2 = pd.DataFrame(data_test_2)


# In[ ]:


#Mean transparency in iRing
mean = []
mean_test = []
mean_test_2 = []
for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))
    mean_test = np.append(mean_test, np.mean(data_df_test[i]))
    mean_test_2 = np.append(mean_test_2, np.mean(data_df_test_2[i]))


# In[ ]:


#Filter data and metadata
mean = mean[mean != -1]
mean_test = mean_test[mean_test != -1]
mean_test_2 = mean_test_2[mean_test_2 != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]


# # Every Fill

# In[ ]:


fill = metadata["fill_num"].unique()
fill= fill[fill != 0]


# In[ ]:


metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]


# In[ ]:


fill_num = metadata_fill.fill_num.unique()
transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)


# In[ ]:


par_1, pcov_1 = curve_fit(transp_func, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)
par_2, pcov_2 = curve_fit(transp_func_2, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)


# In[ ]:


nbin = 600
minimo = 0
massimo = 60
threshold = 30
delta_value = (massimo-minimo)/nbin


# In[ ]:


nEvents = 1000


# # Single fill

# In[ ]:

print("Fill:")
selected_fill = input()


# In[ ]:


selected_metadata = metadata[metadata["fill_num"] == int(selected_fill)]
selected_index = selected_metadata.index.values


# In[ ]:


selected_transp = [mean[i] for i in selected_index]
selected_transp = selected_transp/selected_transp[0]


# In[ ]:


c = ROOT.TCanvas("cc_turn_on0", "", 800, 700)
hist = ROOT.TH1F("real", " ", nbin, minimo, massimo)


# In[ ]:


for ibin in range(0,nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(selected_transp)):
            value_smeared = value*selected_transp[i]
            if value_smeared > threshold:
                hist.Fill(value)


# In[ ]:


hist.Scale(1./(nEvents*np.size(selected_transp)))
  
hist.SetLineWidth(2)
hist.SetLineColor(632)
#hist.Draw("histo")

hist.GetXaxis().SetTitle("Energy [GeV]")
hist.GetYaxis().SetTitle("Efficiency")


# # Third function

# In[ ]:


#c1 = ROOT.TCanvas("cc_turn_on", "", 800, 700)
hist_f1 = ROOT.TH1F("histo", " ", nbin, minimo, massimo)


# In[ ]:


lumi_in_fill = (selected_metadata.lumi_in_fill*(1e-9)).to_numpy()
lumi_inst = (selected_metadata.lumi_inst*(1e-9)).to_numpy()
y_0 = lumi_inst[0]
lumi_inst_0 = np.empty(np.size(lumi_inst))
lumi_inst_0.fill(y_0)


# In[ ]:


transp_smeared = transp_func([lumi_in_fill, lumi_inst, lumi_inst_0], *par_1)


# In[ ]:


for ibin in range(0,nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(transp_smeared)):
            value_smeared = value*transp_smeared[i]
            if value_smeared > threshold :
                hist_f1.Fill(value)


# In[ ]:


hist_f1.Scale(1./(nEvents*np.size(transp_smeared)))
  
hist_f1.SetLineWidth(2)
hist_f1.SetLineColor(603)

hist.Draw("histo")
hist_f1.Draw("histo same")

hist_f1.GetXaxis().SetTitle("Energy [GeV]")
hist_f1.GetYaxis().SetTitle("Efficiency")


# In[ ]:


c.Draw()


# ## Sigmoid

# In[ ]:


#sigmoid = ROOT.TF1("sigmoid", "1/(1+exp(-(x-[0])/[1]))", 0, 60)
#sigmoid.SetParameters(0, threshold)


# In[ ]:


#r = hist.Fit("sigmoid", "s")
#r.Print("V")


# In[ ]:


#fit_1 = hist.GetFunction("sigmoid")
#fit_1.GetProb()


# In[ ]:


#c1.Draw()
#vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
#vertical_line.Draw()
#c1.SaveAs("h_turn_on.png")


# ## Arctan

# In[ ]:


#arctan = ROOT.TF1("arctan","[2]+[3]*atan([1]*(x-[0]))")
#arctan.SetParameters(0, threshold)


# In[ ]:


#r = hist.Fit("arctan", "s")
#r.Print("V")


# In[ ]:


#fit_1 = hist.GetFunction("arctan")
#fit_1.GetProb()


# ## Erf

# In[ ]:


#erf = ROOT.TF1("erf", "[0]*TMath::Erf([2]*(x-[1]))+[0]")
#erf.SetParameters(1, threshold)
#erf.SetParameters(0, 0.5)


# In[ ]:


#r = hist.Fit("erf", "s")
#r.Print("V")


# In[ ]:


#fit_1 = hist.GetFunction("erf")
#fit_1.GetProb()


# # Second function

# In[ ]:


#c2 = ROOT.TCanvas("cc_turn_on2", "", 800, 700)
#hist2 = ROOT.TH1F("f_2", " ", nbin, minimo, massimo)


# In[ ]:


#for ibin in range(0,nbin):
    #value = minimo+(ibin+0.5)*delta_value
    #for iEvent in range(0, nEvents):
        #lumi_in_fill = ROOT.gRandom.Uniform(0.7)
        #lumi_inst = ROOT.gRandom.Uniform(0.0005)
        #y_0 = ROOT.gRandom.Uniform(0.0005)
        #value_smeared = value*(transp_func_2([lumi_in_fill, lumi_inst, y_0], *par_2))
        #if value_smeared > threshold:
            #hist2.Fill(value)


# In[ ]:


#hist2.Scale(1./nEvents)
  
#hist2.SetLineWidth(2)
#hist2.SetLineColor(632)
  
#hist2.Draw("histo")
#hist2.GetXaxis().SetTitle("Energy [GeV]")
#hist2.GetYaxis().SetTitle("Efficiency")


# In[ ]:


#c2.Draw()
#vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
#vertical_line.Draw()
#c2.SaveAs("h_turn_on2.png")


# In[ ]:




