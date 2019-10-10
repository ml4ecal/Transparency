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

fillinfo = namedtuple("fillinfo", ["fill", "beamstatus", "lumi_inst", "lumi_in_fill", "time_in_fill", "time_in_fill_stable"])


# In[2]:


data_EE = np.load("transp_data_EE.npy",mmap_mode="r")
data_EB = np.load("transp_data_EB.npy", mmap_mode="r")


# In[3]:


meta = pd.read_csv("transp_metadata_2017.csv",sep=",").astype({"transp_entry":"int64", "time": "int64", "fill_num":"int64", "time_in_fill":"int64", "time_in_fill_stable":"int64"})
bril = pd.read_csv("lumi_brilcalc_2017.csv", sep=",", comment="#")
bril["run"] = bril.apply(lambda row: int(row["run:fill"].split(":")[0]), axis=1)
bril["fill"] = bril.apply(lambda row: int(row["run:fill"].split(":")[1]), axis=1)


# In[3]:


print(meta.columns)
print(bril.columns)


# ## Starting lumisection

# In[180]:


def gettimes(timestamp, n, t=23):
    o = []
    for i in range(abs(n)):
            o.append(timestamp + i*t)
    if n >0:
        return o
    else:
        o.reverse()
        return o


# Look for the biggest deltaT between measurements

# In[382]:


# meta["time-1"] = meta["time"].shift(1)
# meta["deltaT"] = (meta["time"]- meta["time-1"])


# Cumulative luminosity over fill

# In[7]:


# plt.plot(bril.groupby("fill")["delivered(/ub)"].cumsum())


# ## Cumulative lumi in fill estimation

# In[125]:


bril["lumi_in_fill"] = bril.groupby("fill")["delivered(/ub)"].cumsum()
bril["time_in_fill"] = bril.groupby("fill")["time"].transform(lambda t: t - t.min())
bril["time_in_fill_stable"] = bril[bril.beamstatus=="STABLE BEAMS"].groupby("fill")["time"].transform(lambda t: t - t.min())
bril["time_in_fill_stable"] = bril["time_in_fill_stable"].fillna(0)


# ## Fill start time saving

# In[111]:


def is_in_fill(timestamp):
    a = bril[abs(bril.time - timestamp)<23]
    if len(a):
        a = a.iloc[0]
        return fillinfo(a.fill, a.beamstatus, a["delivered(/ub)"], a.lumi_in_fill, a.time_in_fill, a.time_in_fill_stable)
    else:
        return fillinfo(0, "NOBEAM", 0, 0,0,0)
    
#is_in_fill(starting_time)


# In[112]:


def get_lumi_interval(timestart, timestop):
    return bril[ (bril.time >= timestart) & (bril.time <= timestop)]["delivered(/ub)"].sum()

#get_lumi_interval(starting_time, 1512022951 )


# In[113]:


def get_last_fill_end(timestamp, fill=0):
    df = bril[(bril.time <=timestamp) & (bril.fill != fill)]
    if not df.empty:
        return df.iloc[-1]
    else: 
        return pd.DataFrame()


# In[114]:


def get_fill_timeinterval(fill):
    t = bril[bril.fill== fill].time
    return t.iloc[0], t.iloc[-1]


# In[191]:


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


# In[192]:


previous_time = bril.time.iloc[0]
tot = (bril.time.iloc[-1]-bril.time.iloc[0])//600

for iev, curr_time in enumerate(range(bril.time.iloc[0], bril.time.iloc[-1], 600)):
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


# In[193]:


output_df = pd.DataFrame(outputs)
output_df.to_csv("output_metadata_2017_10min.csv")

# In[4]:


timesteps_df = output_df


# In[5]:


def get_transp_EB(index):
    EB = data_EB[int(index)-1][85, :]
    return EB

def get_transp_EE(index):
    EE = data_EE[int(index)-1]
    return EE


# In[6]:


def get_transp_interval(timestamp):
    last_meas = meta[(meta.time < timestamp)].iloc[-1]
    next_meas = meta[(meta.time > timestamp)]
    if next_meas.empty:
        return last_meas, pd.DataFrame()
    #print("{} {} | {}| x0: {} | Interval diff {:.3f}".format(last_meas.time, next_meas.time,  timestamp,timestamp- last_meas.time, ( next_meas.time - last_meas.time) / 60))
    return last_meas, next_meas.iloc[0]


# In[7]:


def interpolate_transp(x, x0, x1, y0, y1):
    z = (x - x0)*( (y1-y0)/(x1-x0)) + y0
    #print(f"x {x}, x0 {x0}, x1 {x1}, y0 {y0}, y1 {y1} ---> {z}")
    return z


# In[8]:


def get_transp_interpolate(timestamp):
    Z0, Z1 = get_transp_interval(timestamp)
    if Z1.empty:
        return np.array([])
    transp_EB_y0 = get_transp_EB(Z0.transp_entry)
    transp_EB_y1 = get_transp_EB(Z1.transp_entry)
    trans_EB = interpolate_transp(timestamp, Z0.time, Z1.time,  transp_EB_y0, transp_EB_y1)
    #trans_EE =  interpolate_transp(timestamp, meas_interval[0].time, meas_interval[1].time,  transp_EE_y0, transp_EE_y1)
    return trans_EB


# In[9]:


transp_EB_output = []
transp_EE_output = []


# In[10]:


tot = len(timesteps_df)

for iev, row in timesteps_df.iterrows():
    if iev % 100 == 0:
        print(f"{iev}/{tot}")
    
    tEB = get_transp_interpolate(row.time)
    if tEB.size == 0:
        break
    transp_EB_output.append(tEB)
    #transp_EE_output.append(tEE)


# In[11]:


arr = np.array(transp_EB_output)


# In[12]:


np.save("output_transp_EB_ix1_10min.npy", arr)


# In[ ]:




