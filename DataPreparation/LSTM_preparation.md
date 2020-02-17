
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



