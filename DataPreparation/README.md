# Data Preparation 
------------

Some steps are necessary to prepare the data for RNN training. 

## Laser data
The script **LaserDataPreparation.py** is used to extract transparency data for each xtals measured by laser. 
It outpus three files: 
- transp_metadata.csv :  timestamp, laserm etadata, transparency entry
- transp_data_EB: transparency data for EB
- transp_data_EE: transparency data for EE

There are two arrays because they have different geometries and they are saved as arrays with the same geometry of the detector. 

### ECAL geometry
The barrel is represented as 2d array (a rectangle) of sides: 170 eta, 360 phi.
The ieta index goes from 1 to 85 and from -85 to -1 and it is translated to 0-169 indexes with this transformation: 
- if ieta > 0: ieta = ieta + 84
- if ieta < 0: ieta = ieta + 85

The endcaps is represented as 3d array:   2 sides, 100 ix , 100 iy

## Luminosity and fill metadata
These information is taken from brilcalc. 
```
brilcalc lumi -o lumi_file.csv --begin "05/01/17 12:14:02" --end 6500 --tssec -u /ub --byls
```

The script **TimestepsDataPreparation.py** is used to read brilcalc data, and output several metadata with a 10 minutes time interval. 
. 

The output of the script is:
- output_metadata_year_interval.csv:  metadata for interpolated points (lumi, time_in_fill, etc)


## Transparency data interpolation

The laser transparency data has not a regular time interval between points. Usually is ~ 40 minutes during the data taking. 
Now we have fill/lumi metadata for regular (10 minutes) time interval and we want transparency data with the same granularity interpolating the points linearyly. 

To output the interpolated transparency data for a specific point in the detector (ix,iy,iz) let's use the script **output__transp_timestamps.py**, 

```
python output_transp_timestamps.py [-h]
                                   [--transparency-metadata TRANSPARENCY_METADATA]
                                   [--timestamps-metadata TIMESTAMPS_METADATA]
                                   [-o OUTPUT_FILE] [--iz IZ] [--ix IX]
                                   [--iy IY] [-d DATA]

  -h, --help            show this help message and exit
  --transparency-metadata TRANSPARENCY_METADATA
                        Transparency metadata file as given by LaserDataPreparation.py script
  --timestamps-metadata TIMESTAMPS_METADATA
                        Timestamps metadata file as givem by TimestepsDataPreparation.py
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file
  --iz IZ               iz (0 barrel, -1,+1 endcap
  --ix IX               ix (eta barrel, ix endcap)
  --iy IY               iy (phi barrel, iy endcap)
  -d DATA, --data DATA  Data folder
```

This script will created filtered numpy arrays with interpolated transparency data corresponing to the entries in the fill_metadata file. 


