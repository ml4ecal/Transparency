#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams["image.origin"] = 'lower'


data_folder = "/eos/user/d/dvalsecc/www/ECAL/Transparency/data_v1"



data_EE = np.load(f"{data_folder}/transp_data_EE.npy", mmap_mode="r")
data_EB = np.load(f"{data_folder}/transp_data_EB.npy", mmap_mode="r")


# ## Read brilcalc metadata
bril = pd.read_csv("lumi_brilcalc_2017.csv", sep=",", comment="#")
bril["run"] = bril.apply(lambda row: int(row["run:fill"].split(":")[0]), axis=1)
bril["fill"] = bril.apply(lambda row: int(row["run:fill"].split(":")[1]), axis=1)




# Cumulative luminosity over fill

# In[11]:


plt.plot(bril.groupby("fill")["delivered(/ub)"].cumsum())


# ## Cumulative lumi in fill estimation
# Let's calculate cumulative lumi in the fill, and cumulative time in the fill thanks to pandas groupby and transform

# In[12]:


bril.head()


# In[13]:


bril["lumi_in_fill"] = bril.groupby("fill")["delivered(/ub)"].cumsum()
bril["time_in_fill"] = bril.groupby("fill")["time"].transform(lambda t: t - t.min())
bril["time_in_fill_stable"] = bril[bril.beamstatus=="STABLE BEAMS"].groupby("fill")["time"].transform(lambda t: t - t.min())
bril["time_in_fill_stable"] = bril["time_in_fill_stable"].fillna(0)


# In[14]:


bril.head()


# ## Utility functions to query bril information later
# 

# In[15]:


# Useful Namestupls with fill information
fillinfo = namedtuple("fillinfo", ["fill", "beamstatus", "lumi_inst", "lumi_in_fill", "time_in_fill", "time_in_fill_stable"])


# In[18]:


def is_in_fill(timestamp):
    a = bril[abs(bril.time - timestamp)<23]
    if len(a):
        a = a.iloc[0]
        return fillinfo(a.fill, a.beamstatus, a["delivered(/ub)"], a.lumi_in_fill, a.time_in_fill, a.time_in_fill_stable)
    else:
        return fillinfo(0, "NOBEAM", 0, 0,0,0)
    
#is_in_fill(starting_time)


# In[19]:


def get_lumi_interval(timestart, timestop):
    return bril[ (bril.time >= timestart) & (bril.time <= timestop)]["delivered(/ub)"].sum()

#get_lumi_interval(starting_time, 1512022951 )


# In[20]:


def get_last_fill_end(timestamp, fill=0):
    df = bril[(bril.time <=timestamp) & (bril.fill != fill)]
    if not df.empty:
        return df.iloc[-1]
    else: 
        return pd.DataFrame()


# In[21]:


def get_fill_timeinterval(fill):
    t = bril[bril.fill== fill].time
    return t.iloc[0], t.iloc[-1]


# In[23]:


get_fill_timeinterval(6417)


# ## Lumi/fill metadata output
# Let's read bril data to create metadata points every N minutes

# In[24]:


outputs = {
    "in_fill" : [],
    "time": [],
    "lumi_inst": [],
    "lumi_in_fill": [],
    "lumi_since_last_point": [],
    "lumi_last_fill": [],
    "fill_num": [],
    "time_in_fill": [],
    "time_in_fill_stable": [],
    "time_from_last_fill" : [],
    "last_dump_duration" : [],
    "last_fill_duration": [],
}


def add_output(out):
    for k, v in out.items():
        outputs[k].append(v)


# In[26]:


# Interpolation time
time_interval = 600


# In[25]:


previous_time = bril.time.iloc[0]
tot = (bril.time.iloc[-1]-bril.time.iloc[0])//time_interval

for iev, curr_time in enumerate(range(bril.time.iloc[0], bril.time.iloc[-1], time_interval)):
    if iev % 100 == 0:
        print(f"{iev}/{tot}")
    
    fill_info = is_in_fill(curr_time)
    
    last_fill_info = get_last_fill_end(curr_time, fill_info.fill)
    
    if last_fill_info.empty:
        last_fill_end = curr_time
        last_fill_duration = 0
        last_dump_duration = 0
        lumi_last_fill = 0
    else:
        last_fill_end = last_fill_info.time
        last_fill_duration = last_fill_info.time_in_fill
        lumi_last_fill = last_fill_info.lumi_in_fill
        
        if fill_info.fill != 0:
            last_dump_duration =  (curr_time - fill_info.time_in_fill) - last_fill_info.time 
        else:
            last_dump_duration = curr_time - last_fill_info.time 
    
    time_from_last_fill = curr_time - last_fill_end
    lumi_since_last_point = get_lumi_interval(previous_time,curr_time)
    
    out = {
        "in_fill": int(fill_info.fill != 0),
        "time": curr_time,
        "fill_num": fill_info.fill,
        "lumi_inst": fill_info.lumi_inst,
        "lumi_in_fill": fill_info.lumi_in_fill,
        "lumi_since_last_point": lumi_since_last_point,
        "lumi_last_fill": lumi_last_fill,
        
        "time_in_fill": fill_info.time_in_fill,
        "time_in_fill_stable":  fill_info.time_in_fill_stable,
        "time_from_last_fill": time_from_last_fill, 
        "last_dump_duration": last_dump_duration, 
        "last_fill_duration": last_fill_duration
    }
    
    previous_time = curr_time
    add_output(out)

