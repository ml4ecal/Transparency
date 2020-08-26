#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


data_folder=("/home/alessandro/Scrivania/University/ML4ECAL/Transparency/DataPreparation")


# In[ ]:


#Read metadata
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")


# In[ ]:


#Load data
data = np.load(f"{data_folder}/iring_25.npy")
data_df = pd.DataFrame(data)


# In[ ]:


mean = []
for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))


# In[ ]:


#Filter data and metadata
mean = mean[mean != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]


# In[ ]:


#Metadata dei soli fill
metadata_fill = metadata[metadata.in_fill == 1]


# In[ ]:


fill = metadata_fill["fill_num"].unique()


# In[ ]:


mean_fill = []
for f in fill :
    df = metadata_fill[metadata_fill["fill_num"] == f]
    index = df.index.values
    transp_fill = [mean[i] for i in index]
    transp_fill = transp_fill/transp_fill[0]
    mean_fill = np.append(mean_fill, transp_fill)


# # Correlations

# In[ ]:


headers = metadata_fill.columns.tolist()


# In[ ]:


for h in headers:
    correlation_matrix = np.corrcoef(mean_fill, metadata_fill[h])
    correlation_coefficient = correlation_matrix[0][1]
    print(h + " : " + str(correlation_coefficient))


# In[ ]:




