#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import datetime
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from scipy.stats import chisquare


mpl.rcParams['figure.figsize'] = (5,4)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams["image.origin"] = 'lower'
 
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


data_folder=("/home/alessandro/Scrivania/University/ML4ECAL/transparency_ecal/DataPreparation/output_preliminary_plots")


# In[ ]:


#Read metadata
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")


# In[ ]:


#Load data ieta 55
data1 = np.load(f"{data_folder}/output_ix30.npy", allow_pickle=True)
data2 = np.load(f"{data_folder}/output_ix139.npy", allow_pickle=True)

data1_test = np.load(f"{data_folder}/output_iz0_ix140.npy", allow_pickle=True)
data2_test = np.load(f"{data_folder}/output_iz0_ix29.npy", allow_pickle=True)

data_df = pd.DataFrame({'data1':data1, 'data2':data2})
data_df_test = pd.DataFrame({'data1':data1_test, 'data2':data2_test})


# In[ ]:


#Mean transparency in barrel
mean = []
mean_test = []
for i in range(0, np.size(data1)):
    mean = np.append(mean, np.mean(data_df.iloc[i].tolist()))
    mean_test = np.append(mean_test, np.mean(data_df_test.iloc[i].tolist()))


# In[ ]:


#Filter data and metadata
mean = mean[mean != -1]
mean_test = mean_test[mean_test != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]


# In[ ]:


metadata_6371 = metadata[metadata.fill_num == 6371]


# In[ ]:


transp_6371 = mean[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
transp_6371_test = mean_test[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
                   
#Normalizzo
transp_6371 = transp_6371/transp_6371[0]
transp_6371_test = transp_6371_test/transp_6371_test[0]


# In[ ]:


metadata_6287 = metadata[metadata.fill_num == 6287]


# In[ ]:


transp_6287 = mean[metadata_6287.index.values[0]:metadata_6287.index.values[0]+len(metadata_6287.axes[0])]

#Normalizzo
transp_6287 = transp_6287/transp_6287[0]


# In[ ]:


metadata_6156 = metadata[metadata.fill_num == 6156]


# In[ ]:


transp_6156 = mean[metadata_6156.index.values[0]:metadata_6156.index.values[0]+len(metadata_6156.axes[0])]

#Normalizzo
transp_6156 = transp_6156/transp_6156[0]


# In[ ]:


#Load data ieta 20
data3 = np.load(f"{data_folder}/output_iz0_ix104.npy", allow_pickle=True)
data4 = np.load(f"{data_folder}/output_iz0_ix65.npy", allow_pickle=True)

data3_test = np.load(f"{data_folder}/output_iz0_ix105.npy", allow_pickle=True)
data4_test = np.load(f"{data_folder}/output_iz0_ix64.npy", allow_pickle=True)

data2_df = pd.DataFrame({'data3':data3, 'data4':data4})
data2_df_test = pd.DataFrame({'data3':data3_test, 'data4':data4_test})


# In[ ]:


#Mean ieta 20
mean2 = []
mean2_test = []
for i in range(0, np.size(data1)):
    mean2 = np.append(mean2, np.mean(data2_df.iloc[i].tolist()))
    mean2_test = np.append(mean2_test, np.mean(data2_df_test.iloc[i].tolist()))


# In[ ]:


transp_6371_2 = mean[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
transp_6371_2_test = mean_test[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
                   
#Normalizzo
transp_6371_2 = transp_6371/transp_6371[0]
transp_6371_2_test = transp_6371_test/transp_6371_test[0]


# ## Fit esponenziali $|i\eta|$ 55

# In[ ]:


def fit_func(x, a, b, c):
    return a*np.exp(-x*b)+ (1-a)*np.exp(x*c)


# In[ ]:


params = curve_fit(fit_func, metadata_6371.lumi_in_fill*(1e-9), transp_6371)
params[0]


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2]), label='a=%5.3f,  b=%5.3f, c=%5.3f' % tuple(params[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371, ".r-", markersize=2, linewidth=0.75, label='fill 3671')
plt.legend(title='i$\eta$ 55')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx} + (1-a)e^{cx}$')
plt.savefig('fit_exp_exp')


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2]), label='a=%5.3f,  b=%5.3f, 3 c=%5.3f' % tuple(params[0])) 
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_test, ".r-", markersize=2, linewidth=0.75, label='fill 3671')
plt.legend(title='i$\eta$ 56')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean trannsparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.savefig('fit_exp_exp_test')


# In[ ]:


chi2 = chisquare(transp_6371_test, fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2]))
chi2


# ## Fit esponeneziale + parabola $|i\eta|$ 55

# In[ ]:


def fit_func_2(x, a, b, c, d, e):
    return a*np.exp(-x*b)+c*x**2+d*x+e


# In[ ]:


params_2 = curve_fit(fit_func_2, metadata_6371.lumi_in_fill*(1e-9), transp_6371)
params_2[0]


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func_2(metadata_6371.lumi_in_fill*(1e-9), params_2[0][0], params_2[0][1], params_2[0][2], params_2[0][3], params_2[0][4]), label='a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(params_2[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='i$\eta$ 55', prop={'size':8})
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean trannsparency')
plt.title('ae$^{-bx}$ + cx$^2$ + dx + e')
plt.savefig('fit_exp_parabola')


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func_2(metadata_6371.lumi_in_fill*(1e-9), params_2[0][0], params_2[0][1], params_2[0][2], params_2[0][3], params_2[0][4]), label= 'a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(params_2[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_test, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='i$\eta$ 56', prop={'size':8})
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('ae$^{-bx}$ + cx$^2$ + dx + e')
plt.savefig('fit_exp_parabola_test')


# ## Fit esponenziali $|i\eta|$ 20

# In[ ]:


par = curve_fit(fit_func, metadata_6371.lumi_in_fill*(1e-9), transp_6371_2)
par[0]


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_2, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='i$\eta$ 20')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.savefig('fit_exp_exp')


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_2_test, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='i$\eta$ 20')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.savefig('fit_exp_exp_EB2_test')


# ## Fitting fill 6156

# In[ ]:


par = curve_fit(fit_func, metadata_6156.lumi_in_fill*(1e-9), transp_6156)
par[0]


# In[ ]:


plt.plot(metadata_6156.lumi_in_fill*(1e-9), fit_func(metadata_6156.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6156.lumi_in_fill*(1e-9), transp_6156, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 55'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6156')


# ## Fitting fill 6156 test

# In[ ]:


transp_6156_test = mean_test[metadata_6156.index.values[0]:metadata_6156.index.values[0]+len(metadata_6156.axes[0])]
transp_6156_test = transp_6156_test/transp_6156_test[0]


# In[ ]:


plt.plot(metadata_6156.lumi_in_fill*(1e-9), fit_func(metadata_6156.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6156.lumi_in_fill*(1e-9), transp_6156_test, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 56'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6156_test')


# # Fitting fill 6287

# In[ ]:


par = curve_fit(fit_func, metadata_6287.lumi_in_fill*(1e-9), transp_6287)
par[0]


# In[ ]:


plt.plot(metadata_6287.lumi_in_fill*(1e-9), fit_func(metadata_6287.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6287.lumi_in_fill*(1e-9), transp_6287, ".r-", markersize=2, linewidth=0.75, label='fill 6287')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 55'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6287')


# ## Fitting fill 6287 test

# In[ ]:


transp_6287_test = mean_test[metadata_6287.index.values[0]:metadata_6287.index.values[0]+len(metadata_6287.axes[0])]


# In[ ]:


transp_6287_test = transp_6287_test/transp_6287_test[0]


# In[ ]:


plt.plot(metadata_6287.lumi_in_fill*(1e-9), fit_func(metadata_6287.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6287.lumi_in_fill*(1e-9), transp_6287_test, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 56'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6287_test')


# # Fitting fill 6026

# In[ ]:


metadata_6026 = metadata[metadata.fill_num == 6026]


# In[ ]:


transp_6026 = mean[metadata_6026.index.values[0]:metadata_6026.index.values[0]+len(metadata_6026.axes[0])]


# In[ ]:


transp_6026 = transp_6026/transp_6026[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6026.lumi_in_fill*(1e-9), transp_6026)
par[0]


# In[ ]:


plt.plot(metadata_6026.lumi_in_fill*(1e-9), fit_func(metadata_6026.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6026.lumi_in_fill*(1e-9), transp_6026, ".r-", markersize=2, linewidth=0.75, label='fill 6026')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 55'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6026')


# ## Fitting fill 6026 test

# In[ ]:


transp_6026_test = mean_test[metadata_6026.index.values[0]:metadata_6026.index.values[0]+len(metadata_6026.axes[0])]


# In[ ]:


transp_6026_test = transp_6026_test/transp_6026_test[0]


# In[ ]:


plt.plot(metadata_6026.lumi_in_fill*(1e-9), fit_func(metadata_6026.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6026.lumi_in_fill*(1e-9), transp_6026_test, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 56'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6026_test')


# In[ ]:


chi = chisquare(transp_6026_test, fit_func(metadata_6026.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]))
chi


# # Fitting fill 6191

# In[ ]:


metadata_6191 = metadata[metadata.fill_num == 6191]


# In[ ]:


transp_6191 = mean[metadata_6191.index.values[0]:metadata_6191.index.values[0]+len(metadata_6191.axes[0])]


# In[ ]:


transp_6191 = transp_6191/transp_6191[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6191.lumi_in_fill*(1e-9), transp_6191)
par[0]


# In[ ]:


plt.plot(metadata_6191.lumi_in_fill*(1e-9), fit_func(metadata_6191.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6191.lumi_in_fill*(1e-9), transp_6191, ".r-", markersize=2, linewidth=0.75, label='fill 6191')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 55'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6191')


# ## Fitting fill 6191 test

# In[ ]:


transp_6191_test = mean_test[metadata_6191.index.values[0]:metadata_6191.index.values[0]+len(metadata_6191.axes[0])]


# In[ ]:


transp_6191_test = transp_6191_test/transp_6191_test[0]


# In[ ]:


plt.plot(metadata_6191.lumi_in_fill*(1e-9), fit_func(metadata_6191.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6191.lumi_in_fill*(1e-9), transp_6191_test, ".r-", markersize=2, linewidth=0.75, label='fill 6191')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 56'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6191_test')


# # Fitting fill 6314

# In[ ]:


metadata_6314 = metadata[metadata.fill_num == 6314]


# In[ ]:


transp_6314 = mean[metadata_6314.index.values[0]:metadata_6314.index.values[0]+len(metadata_6314.axes[0])]


# In[ ]:


transp_6314 = transp_6314/transp_6314[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6314.lumi_in_fill*(1e-9), transp_6314)


# In[ ]:


plt.plot(metadata_6314.lumi_in_fill*(1e-9), fit_func(metadata_6314.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6314.lumi_in_fill*(1e-9), transp_6314, ".r-", markersize=2, linewidth=0.75, label='fill 6314')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 55'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp_exp_6314')


# ## Fitting fill 6314 test

# In[ ]:


transp_6314_test = mean_test[metadata_6314.index.values[0]:metadata_6314.index.values[0]+len(metadata_6314.axes[0])]


# In[ ]:


transp_6314_test = transp_6314_test/transp_6314_test[0]


# In[ ]:


plt.plot(metadata_6314.lumi_in_fill*(1e-9), fit_func(metadata_6314.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6314.lumi_in_fill*(1e-9), transp_6314_test, ".r-", markersize=2, linewidth=0.75, label='fill 6414')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title=('i$\eta$ 56'))
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized Mean Transparency')
plt.savefig('fit_exp-exp_6314_test')

