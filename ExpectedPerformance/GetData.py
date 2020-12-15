#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


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


# In[ ]:


fill = metadata["fill_num"].unique()
fill = fill[fill != 0]


# In[ ]:


metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]


# In[ ]:


#Get transparency data
fill_num = metadata_fill.fill_num.unique()
transp_fill_26 = []
transp_fill_25 = []
transp_fill_24 = []

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean_test[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill_26 = np.append(transp_fill_26, transp)
    transp = [mean[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill_25 = np.append(transp_fill_25, transp)
    transp = [mean_test_2[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill_24 = np.append(transp_fill_24, transp)


# In[ ]:


# Save .txt files
output = np.vstack([transp_fill_24, metadata_fill.lumi_inst*(1e-9), metadata_fill.lumi_in_fill*(1e-9), metadata_fill.time]).T
np.savetxt("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/ExpectedPerformance/iring24.txt", output, fmt="%s %s %s %s")

output = np.vstack([transp_fill_25, metadata_fill.lumi_inst*(1e-9), metadata_fill.lumi_in_fill*(1e-9), metadata_fill.time]).T
np.savetxt("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/ExpectedPerformance/iring25.txt", output, fmt="%s %s %s %s")

output = np.vstack([transp_fill_26, metadata_fill.lumi_inst*(1e-9), metadata_fill.lumi_in_fill*(1e-9), metadata_fill.time]).T
np.savetxt("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/ExpectedPerformance/iring26.txt", output, fmt="%s %s %s %s")


# In[ ]:




