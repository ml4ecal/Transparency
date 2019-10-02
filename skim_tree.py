import sys
import os
import argparse
from itertools import islice
import ROOT as R
import pandas as pd
import numpy as np  
from tqdm import tqdm

df_lumi = pd.read_csv("insta_lumi_brilcalc_2017.csv", sep=",")

def get_lumi(timestamp):
    a=  df_lumi[(df_lumi.time>=timestamp) & (df_lumi.time < timestamp+23)]
    if len(a):
        return True, a["delivered(/ub)"].iloc[0]
    else:
        return False, 0.

f = R.TFile("data/BlueLaser_2017_rereco_v2_newformat.root")
tree = f.Get("ntu")

pbar = tqdm(total=tree.GetEntries())


outputs = {
    "transp" : [],
    "time": [],
    "lumi": [],
    "intlumi": [],
    "last_fill_intlumi": [],
    "bfield": [],
    "fill_num": [],
    "fill_time_start": [],
    "fill_time_stable_start": [],
    "time_in_fill": [],
    "time_in_fill_stable": [],
    "time_from_last_fill" : [],
    "last_dump_duration" : [],
    "last_fill_duration": [],
    "run_in_fill" : [],
    "deltaT" : [], 
    "in_fill" : []
}


if len(sys.argv)>2:
    nevent = int(sys.argv[2])
    if len(sys.argv) > 3:
        nevent2 = int(sys.argv[3])
    else:
        nevent2 = nevent+1
    tree = islice(tree, nevent, nevent2)
 
latest = {}

def add_output(out):
    for k, v in out.items():
        outputs[k].append(v)

last_fill_dump_time = 0


current_fill_time = 0
last_fill_lumi = 0
current_fill_lumi = 0

last_fill_duration = 0.

current_fill_start_time = 0.
last_fill_start_time = 0

for iev, event in enumerate(tree):
    pbar.update()

    out = {}
    out["time"] = event.time[0]
    isinfill, out["lumi"] = get_lumi(out["time"])
    out["bfield"] = event.bfield
    out["run_in_fill"] = event.run_num_infill
    out["fill_num"] = event.fill_num
    out["fill_time_start"] = event.fill_time 
    out["fill_time_stable_start"] = event.fill_time_stablebeam
    out["in_fill"] = int(isinfill)

    if iev == 0:
        current_fill_time = out["time"]
        latest = out
        continue  
    
    if isinfill:
        # Save the current time if it is in fill
        current_fill_time = out["time"]
        out["time_in_fill"] = event.time[0] - event.fill_time
        time_infill_stable = event.time[0] - event.fill_time_stablebeam
        if time_infill_stable > 0: 
            out["time_in_fill_stable"] = time_infill_stable
        else: 
            out["time_in_fill_stable"] = 0
        # Lenght of the last dump
        out["last_dump_duration"] = out["fill_time_start"] - last_fill_dump_time
    else:
        # Switch the last current_fill_time as dump time
        last_fill_dump_time = current_fill_time
        

        out["time_in_fill"] = 0.
        out["time_in_fill_stable"] = 0.
        out["last_dump_duration"] = out["time"] - last_fill_dump_time
        
    
    out["time_from_last_fill"] = out["time"] - last_fill_dump_time
     

    # Using previous point information 
    out["deltaT"] = out["time"] - latest["time"]

    # integrate luminosity of current fill
    if not isinfill:
        out["intlumi"] = 0.
    elif isinfill and not latest["in_fill"]:
        # Start the new fill
        out["intlumi"] = out["lumi"]
        current_fill_start_time = out["time"]
    else:
        out["intlumi"] = latest["intlumi"] + out["lumi"]
    
    
    if isinfill:
        current_fill_lumi = out["intlumi"]
    else:
        last_fill_lumi = current_fill_lumi

    out["last_fill_intlumi"] = last_fill_lumi
    

   # transp = []
    # for iz, ix, iy, tr in zip(event.iz, event.ix, event.iy, event.nrv):
    #     transp.append((iz,ix,iy, tr))
    # out["transp"] = transp  

    add_output(out)
    latest = out

    print(out)



