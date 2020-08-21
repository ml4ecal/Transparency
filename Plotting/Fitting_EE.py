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


data_folder=("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/DataPreparation")


# In[ ]:


#Read metadata
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")


# In[ ]:


#Load data
data = np.load(f"{data_folder}/iring_25.npy")
data_test = np.load(f"{data_folder}/iring_26.npy")

data_df = pd.DataFrame(data)
data_df_test = pd.DataFrame(data_test)


# In[ ]:


#Mean transparency in iRing
mean = []
mean_test = []
for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))
    mean_test = np.append(mean_test, np.mean(data_df_test[i]))


# In[ ]:


#Filter data and metadata
mean = mean[mean != -1]
mean_test = mean_test[mean_test != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]


# In[ ]:


metadata_6371 = metadata[metadata.fill_num == 6371]


# In[ ]:


transp_6371 = mean[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
                   
#Normalizzo
transp_6371 = transp_6371/transp_6371[0]


# In[ ]:


metadata_6156 = metadata[metadata.fill_num == 6156]


# In[ ]:


transp_6156 = mean[metadata_6156.index.values[0]:metadata_6156.index.values[0]+len(metadata_6156.axes[0])]

#Normalizzo
transp_6156 = transp_6156/transp_6156[0]


# In[ ]:


transp_6371_test = mean_test[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]

#Normalizzo
transp_6371_test = transp_6371_test/transp_6371_test[0]


# ## Fit esponenziali

# In[ ]:


def fit_func(x, a, b, c):
    return a*np.exp(-x*b)+ (1-a)*np.exp(x*c)


# In[ ]:


params = curve_fit(fit_func, metadata_6371.lumi_in_fill*(1e-9), transp_6371)
params[0]


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371, ".r-", markersize=2, linewidth=0.75, label='fill 3671')
plt.legend(title='iRing 25')
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.tick_params(labelsize=7)
plt.title('$ae^{-bx}+(1-a)e^{dx}$')
plt.savefig('fit_exp_exp')


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_test, ".r-", markersize=2, linewidth=0.75, label='fill 3671')
plt.legend(title='iRing 26')
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.tick_params(labelsize=7)
plt.title('$ae^{-bx}+(1-a)e^{dx}$')
plt.savefig('fit_exp_exp_test')


# In[ ]:


print(chisquare(transp_6371_test, fit_func(metadata_6371.lumi_in_fill*(1e-9), params[0][0], params[0][1], params[0][2])))


# ## Fit esponeneziale + parabola

# In[ ]:


def fit_func_2(x, a, b, c, d, e):
    return a*np.exp(-x*b)+c*x**2+d*x+e


# In[ ]:


params_2 = curve_fit(fit_func_2, metadata_6371.lumi_in_fill*(1e-9), transp_6371)
params_2[0]


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func_2(metadata_6371.lumi_in_fill*(1e-9), params_2[0][0], params_2[0][1], params_2[0][2], params_2[0][3], params_2[0][4]), label='a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %tuple(params_2[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='iRing 25', prop={'size':8})
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.tick_params(labelsize=7)
plt.title('$ae^{-bx}+cx^2+dx+e$')
plt.savefig('fit_exp_parabola')


# In[ ]:


plt.plot(metadata_6371.lumi_in_fill*(1e-9), fit_func_2(metadata_6371.lumi_in_fill*(1e-9), params_2[0][0], params_2[0][1], params_2[0][2], params_2[0][3], params_2[0][4]), label='a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %tuple(params_2[0]))
plt.plot(metadata_6371.lumi_in_fill*(1e-9), transp_6371_test, ".r-", markersize=2, linewidth=0.75, label='fill 6371')
plt.legend(title='iRing 26', prop={'size':8})
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.tick_params(labelsize=7)
plt.title('$ae^{-bx}+cx^2+dx+e$')
plt.savefig('fit_exp_parabola_test')


# In[ ]:


print(chisquare(transp_6371_test, fit_func_2(metadata_6371.lumi_in_fill*(1e-9), params_2[0][0], params_2[0][1], params_2[0][2], params_2[0][3], params_2[0][4])))


# # Fitting fill 6156

# In[ ]:


metadata_6156 = metadata[metadata.fill_num == 6156]


# In[ ]:


transp_6156 = mean[metadata_6156.index.values[0]:metadata_6156.index.values[0]+len(metadata_6156.axes[0])]
transp_6156 = transp_6156/transp_6156[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6156.lumi_in_fill*(1e-9), transp_6156)
par[0]


# In[ ]:


plt.plot(metadata_6156.lumi_in_fill*(1e-9), fit_func(metadata_6156.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6156.lumi_in_fill*(1e-9), transp_6156, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 25')
plt.savefig('fit_exp_exp_6156')


# ## Fitting fill 6156 test

# In[ ]:


transp_6156_test = mean_test[metadata_6156.index.values[0]:metadata_6156.index.values[0]+len(metadata_6156.axes[0])]
transp_6156_test = transp_6156_test/transp_6156_test[0]


# In[ ]:


plt.plot(metadata_6156.lumi_in_fill*(1e-9), fit_func(metadata_6156.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6156.lumi_in_fill*(1e-9), transp_6156_test, ".r-", markersize=2, linewidth=0.75, label='fill 6156')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 26')
plt.savefig('fit_exp_exp_6156_test')


# # Fitting fill 6287

# In[ ]:


metadata_6287 = metadata[metadata.fill_num == 6287]


# In[ ]:


transp_6287 = mean[metadata_6287.index.values[0]:metadata_6287.index.values[0]+len(metadata_6287.axes[0])]
transp_6287 = transp_6287/transp_6287[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6287.lumi_in_fill*(1e-9), transp_6287)
par[0]


# In[ ]:


plt.plot(metadata_6287.lumi_in_fill*(1e-9), fit_func(metadata_6287.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6287.lumi_in_fill*(1e-9), transp_6287, ".r-", markersize=2, linewidth=0.75, label='fill 6287')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 25')
plt.savefig('fit_exp_exp_6287')


# ## Fitting fill 6287 test

# In[ ]:


transp_6287_test = mean_test[metadata_6287.index.values[0]:metadata_6287.index.values[0]+len(metadata_6287.axes[0])]
transp_6287_test = transp_6287_test/transp_6287_test[0]


# In[ ]:


plt.plot(metadata_6287.lumi_in_fill*(1e-9), fit_func(metadata_6287.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6287.lumi_in_fill*(1e-9), transp_6287_test, ".r-", markersize=2, linewidth=0.75, label='fill 6287')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 26')
plt.savefig('fit_exp_exp_6287_test')


# # Fitting fill 6026

# In[ ]:


metadata_6026 = metadata[metadata.fill_num == 6026]


# In[ ]:


transp_6026 = mean[metadata_6026.index.values[0]:metadata_6026.index.values[0]+len(metadata_6026.axes[0])]
transp_6026 = transp_6026/transp_6026[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6026.lumi_in_fill*(1e-9), transp_6026)
par[0]


# In[ ]:


plt.plot(metadata_6026.lumi_in_fill*(1e-9), fit_func(metadata_6026.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6026.lumi_in_fill*(1e-9), transp_6026, ".r-", markersize=2, linewidth=0.75, label='fill 6026')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 25')
plt.savefig('fit_exp_exp_6026')


# ## Fitting fill 6026 test

# In[ ]:


transp_6026_test = mean_test[metadata_6026.index.values[0]:metadata_6026.index.values[0]+len(metadata_6026.axes[0])]
transp_6026_test = transp_6026_test/transp_6026_test[0]


# In[ ]:


plt.plot(metadata_6026.lumi_in_fill*(1e-9), fit_func(metadata_6026.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6026.lumi_in_fill*(1e-9), transp_6026_test, ".r-", markersize=2, linewidth=0.75, label='fill 6026')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 26')
plt.savefig('fit_exp_exp_6026_test')


# # Fitting fill 6191

# In[ ]:


metadata_6191 = metadata[metadata.fill_num == 6191]


# In[ ]:


transp_6191 = mean[metadata_6191.index.values[0]:metadata_6191.index.values[0]+len(metadata_6191.axes[0])]
transp_6191 = transp_6191/transp_6191[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6191.lumi_in_fill*(1e-9), transp_6191)


# In[ ]:


plt.plot(metadata_6191.lumi_in_fill*(1e-9), fit_func(metadata_6191.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6191.lumi_in_fill*(1e-9), transp_6191, ".r-", markersize=2, linewidth=0.75, label='fill 6191')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 25')
plt.savefig('fit_exp_exp_6191')


# ## Fitting fill 6191 test

# In[ ]:


transp_6191_test = mean_test[metadata_6191.index.values[0]:metadata_6191.index.values[0]+len(metadata_6191.axes[0])]
transp_6191_test = transp_6191_test/transp_6191_test[0]


# In[ ]:


plt.plot(metadata_6191.lumi_in_fill*(1e-9), fit_func(metadata_6191.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6191.lumi_in_fill*(1e-9), transp_6191_test, ".r-", markersize=2, linewidth=0.75, label='fill 6191')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 26')
plt.savefig('fit_exp_exp_6191_test')


# # Fitting fill 6314

# In[ ]:


metadata_6314 = metadata[metadata.fill_num == 6314]


# In[ ]:


transp_6314 = mean[metadata_6314.index.values[0]:metadata_6314.index.values[0]+len(metadata_6314.axes[0])]
transp_6314 = transp_6314/transp_6314[0]


# In[ ]:


par = curve_fit(fit_func, metadata_6314.lumi_in_fill*(1e-9), transp_6314)
par[0]


# In[ ]:


plt.plot(metadata_6314.lumi_in_fill*(1e-9), fit_func(metadata_6314.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6314.lumi_in_fill*(1e-9), transp_6314, ".r-", markersize=2, linewidth=0.75, label='fill 6314')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 25')
plt.savefig('fit_exp_exp_6314')


# ## Fitting fill 6314 test

# In[ ]:


transp_6314_test = mean_test[metadata_6314.index.values[0]:metadata_6314.index.values[0]+len(metadata_6314.axes[0])]
transp_6314_test = transp_6314_test/transp_6314_test[0]


# In[ ]:


plt.plot(metadata_6314.lumi_in_fill*(1e-9), fit_func(metadata_6314.lumi_in_fill*(1e-9), par[0][0], par[0][1], par[0][2]), label='a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par[0]))
plt.plot(metadata_6314.lumi_in_fill*(1e-9), transp_6314_test, ".r-", markersize=2, linewidth=0.75, label='fill 6314')
plt.tick_params(labelsize=7)
plt.xlabel('Lumi in fill [fb$^{-1}$]')
plt.ylabel('Normalized mean transparency')
plt.title('$ae^{-bx}+(1-a)e^{cx}$')
plt.legend(title='iRing 26')
plt.savefig('fit_exp_exp_6314_test')


# In[ ]:




