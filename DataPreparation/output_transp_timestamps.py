#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--transparency-metadata", help="Transparency metadata file", type=str)
parser.add_argument("--timestamps-metadata", help="Timestamps metadata file", type=str)
parser.add_argument("-o","--output-file", help="Output file", type=str)
parser.add_argument("--iz", help="iz (0 barrel, -1,+1 endcap", type=int)
parser.add_argument("--ix", help="ix (eta barrel, ix endcap)", type=int)
parser.add_argument("--iy", help="iy (phi barrel, iy endcap)", type=int)
parser.add_argument("-d","--data", help="Data folder")
args = parser.parse_args()



data_EE = np.load(f"{args.data}/transp_data_EE.npy", mmap_mode="r")
data_EB = np.load(f"{args.data}/transp_data_EB.npy", mmap_mode="r")


# ## Read brilcalc metadata

meta = pd.read_csv(args.transparency_metadata,sep=",").astype(
            {"transp_entry":"int64", "time": "int64", "fill_num":"int64", "time_in_fill":"int64", "time_in_fill_stable":"int64"})


timestamps_df = pd.read_csv(args.timestamps_metadata, sep=",", comment="#")



def get_transp_EB(index):
    EB = data_EB[int(index)-1][args.ix,args.iy]
    return EB

def get_transp_EE(index, z):
    EE = data_EE[int(index)-1][z][args.ix,args.iy]
    return EE

def get_transp_interval(timestamp):
    last_meas = meta[(meta.time < timestamp)].iloc[-1]
    next_meas = meta[(meta.time > timestamp)]
    if next_meas.empty:
        return last_meas, pd.DataFrame()
    #print("{} {} | {}| x0: {} | Interval diff {:.3f}".format(last_meas.time, next_meas.time,  timestamp,timestamp- last_meas.time, ( next_meas.time - last_meas.time) / 60))
    return last_meas, next_meas.iloc[0]


def interpolate_transp(x, x0, x1, y0, y1):
    z = (x - x0)*( (y1-y0)/(x1-x0)) + y0
    #print(f"x {x}, x0 {x0}, x1 {x1}, y0 {y0}, y1 {y1} ---> {z}")
    return z


def get_transp_interpolate(timestamp):
    Z0, Z1 = get_transp_interval(timestamp)
    if Z1.empty:
        return np.array([])

    if args.iz == 0:
        transp_EB_y0 = get_transp_EB(Z0.transp_entry)
        transp_EB_y1 = get_transp_EB(Z1.transp_entry)
        trans_EB = interpolate_transp(timestamp, Z0.time, Z1.time,  transp_EB_y0, transp_EB_y1)
        return trans_EB
    
    if args.iz == 1:
        transp_EE_y0 = get_transp_EE(Z0.transp_entry, 0)
        transp_EE_y1 = get_transp_EE(Z1.transp_entry, 0)
        trans_EE =  interpolate_transp(timestamp,  Z0.time, Z1.time,  transp_EE_y0, transp_EE_y1)
        return trans_EE

    if args.iz == -1:
        transp_EE_y0 = get_transp_EE(Z0.transp_entry, 1)
        transp_EE_y1 = get_transp_EE(Z1.transp_entry, 1)
        trans_EE =  interpolate_transp(timestamp,  Z0.time, Z1.time,  transp_EE_y0, transp_EE_y1)
        return trans_EE
   


transp_output = []
tot = len(timestamps_df)

for iev, row in timestamps_df.iterrows():
    if iev % 100 == 0:
        print(f"{iev}/{tot}")
    
    tran = get_transp_interpolate(row.time)
    
    if tran.shape == 0:
        break
    transp_output.append(tran)


np.save(args.output_file,  np.array(transp_output))






