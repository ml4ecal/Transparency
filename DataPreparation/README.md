# Data Preparation 
------------

Some steps are necessary to prepare the data for RNN training. 

## Laser data
The script **LaserDataPreparation.py** is used to extract transparency data for each xtals measured by laser. 
It outpus three files: 
- transp_metadata.csv :  timestamp, lasermetadata
- transp_data_EB: transparency data for EB
- transp_data_EE: transparency data for EE

There are two arrays because they have different geometries and they are saved as arrays with the same geometry of the detector. 

Inputs from: 

    /eos/cms/store/group/dpg_ecal/comm_ecal/pedestals_gainratio/BlueLaser_2011-2018_newformat.root 
    (copy here) http://dvalsecc.web.cern.ch/dvalsecc/ECAL/Transparency/data_v1/
    
Output:

    http://dvalsecc.web.cern.ch/dvalsecc/ECAL/Transparency/
    

## Luminosity and fill metadata
These information is taken from brilcalc. 

```
brilcalc lumi -o lumi_file.csv --begin "05/01/17 12:14:02" --end 6500 --tssec -u /ub --byls
```

At this point it is useful to interpolate the laser data with regular timestamps. 
The script **TimestepsDataPreparation.py** is used to read brilcalc data, laser metadata and interpolate
the transparency data at constant timestamp. 
Linear interpolation is used between points. 

The output of the script is:
- output_metadata.csv:  metadata for interpolated points (lumi, time_in_fill, etc)
- transp_data.npy: numpy transparency data for the interpolated points

## Preparation for LSTM network

Now we have all the data but we have to choose the best format for the NN training. 

The idea is to use the last N measurement (also not regular in time) and to predict the next M measurement at costant timestamps. 
N.B: For the moment we are just using the interpolated measurements 1 every 40 minutes instead of real measurements interval..

The script **LSTMInputPreparation.py** prepares numpy arrays suitable for training. 
It reads the output of the previous steps:

Window configuration: 
- Start from the timestamp of 1 measurement
- Read 15 measurements before 
- Read 48 interpolated measures (10 minutes interval) ahead

Then the window is moved and another set of trasparencies and metadata is prepared. 

We want to try the encoder-decoder (or sequence-2-sequence) network:  (https://arxiv.org/abs/1409.3215)
- The *encoder* input will be fill metadata and measured transparency
- The *decoder* inputs will be fill metadata and the output the predicted transparency
- The *decoder* target will be true (interpolated) transparencies



