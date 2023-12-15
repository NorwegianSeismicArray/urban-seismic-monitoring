# Copyright 2023 Andreas Koehler, MIT license

"""
Code for outlier detection

"""

import numpy as np

from train_AutoEncoder import CAE
from obspy.core import UTCDateTime,Stream,Trace
from utils import create_waveforms, map_station_to_shake
from obspy.signal.trigger import coincidence_trigger
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler, TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from obspy.signal.util import next_pow_2

# Data is not on the Norwegian EIDA node yet
# Code will only run fro now if you adapt to your onw data
#from obspy.clients.fdsn import Client
#client = Client("UIB-NORSAR")
from obspy.clients.filesystem.sds import Client
client = Client('/adhoc/seed/myshake/')

def data_to_trace(data,samp_rate,starttime,label,channel):
    tr=Trace()
    tr.stats.sampling_rate=samp_rate
    tr.stats.starttime = starttime
    tr.stats.station = label
    tr.stats.channel = channel
    tr.data = data
    return tr

def main(station, station_org, start, duration, model, winlen, flow, fhigh, stalta_threshold = 10.0, corr_threshold = 0.925, plot_level = False):

    end = start + duration
    timeseries_size = next_pow_2(2*winlen*fhigh)
    print("Using time window length of",timeseries_size,"seconds.")

    print('Loading data from:',start,'to:',end)
    st = client.get_waveforms('*',station,'*','*', start, end)
    st.merge()

    for tr in st:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled(0.0)
    st.detrend('demean')
    st.taper(max_percentage=0.01)
    if flow is not None:
        st.filter('highpass', freq=flow)
    if fhigh is not None:
        st.filter('lowpass', freq=fhigh)

    trig = coincidence_trigger("recstalta", stalta_threshold, 1, st, 3, sta=0.5, lta=10, details = True)
    print('Number of sta/lta triggered outliers:', len(trig))
    if plot_level == 1 :
        for trigger in trig:
            print(trigger['time'],np.mean(trigger['cft_peaks']))
            st.slice(starttime=trigger['time']-30,endtime=trigger['time']+30).plot()

    corr_trigger={}
    corr_trigger['time']=[]
    corr_trigger['duration']=[]
    corr_trigger['correlation']=[]
    corr_trigger['stalta']=[]

    for trigger in trig:
        print('Loading data for trigger:',trigger['time'])
        # start 1.25 time windows before trigger time and stop 1.75 after
        wstart = trigger['time']-winlen/4-winlen
        wend = trigger['time']+winlen*3/4+winlen
        try :
            print(wstart, wend, [station],winlen,flow,fhigh)
            X, W, _ = create_waveforms(wstart, wend, [station],
                         winlen=winlen,
                         overlap=0,
                         bandpass=(flow,fhigh))

            X = TimeSeriesResampler(timeseries_size).fit_transform(X)
            #X = TimeSeriesScalerMeanVariance().fit_transform(X)
            X = TimeSeriesScalerMinMax().fit_transform(X)
        except ValueError :
            continue
        pred = model.predict(X, verbose=1)

        # create a stream with data and reconstructed data
        st=Stream()
        samp_rate = len(np.concatenate(X))/(winlen*3)
        st += data_to_trace(np.transpose(np.concatenate(X))[0],samp_rate,wstart,'X','1')
        st += data_to_trace(np.transpose(np.concatenate(X))[1],samp_rate,wstart,'X','2')
        st += data_to_trace(np.transpose(np.concatenate(X))[2],samp_rate,wstart,'X','3')
        st += data_to_trace(np.transpose(np.concatenate(pred))[0],samp_rate,wstart,'P','1')
        st += data_to_trace(np.transpose(np.concatenate(pred))[1],samp_rate,wstart,'P','2')
        st += data_to_trace(np.transpose(np.concatenate(pred))[2],samp_rate,wstart,'P','3')

        x=st.copy().select(station='X')
        p=st.copy().select(station='P')
        corr_len = timeseries_size
        # for correlation coefficient we cut the traces again to original time window length of autoencoder
        corr_test = np.corrcoef(x[0].data[corr_len:2*corr_len],p[0].data[corr_len:2*corr_len])[0][1]
        corr_test += np.corrcoef(x[1].data[corr_len:2*corr_len],p[1].data[corr_len:2*corr_len])[0][1]
        corr_test += np.corrcoef(x[2].data[corr_len:2*corr_len],p[2].data[corr_len:2*corr_len])[0][1]
        corr_test /= 3
        if plot_level > 0 and corr_test < corr_threshold :
            print("Correlation trigger:",trigger,corr_test)
            fig = plt.figure(figsize=(10,5))
            xa=fig.add_subplot(111)
            times=np.arange(0,len(x[0].data[corr_len:-corr_len]),1)/samp_rate
            xa.plot(times,x[0].data[corr_len:-corr_len]-2)
            xa.plot(times,p[0].data[corr_len:-corr_len]-2)
            plt.gca().set_prop_cycle(None)
            xa.plot(times,x[1].data[corr_len:-corr_len])
            xa.plot(times,p[1].data[corr_len:-corr_len])
            plt.gca().set_prop_cycle(None)
            xa.plot(times,x[2].data[corr_len:-corr_len]+2)
            xa.plot(times,p[2].data[corr_len:-corr_len]+2)
            plt.xlabel('Time (s)')
            plt.legend(['Observed','Reconstructed'])
            plt.yticks([])
            plt.title(str(trigger['time'])[:19])
            plt.show()
        if corr_test < corr_threshold :
            corr_trigger['time'].append(trigger['time'])
            corr_trigger['duration'].append(trigger['duration'])
            corr_trigger['correlation'].append(corr_test)
            corr_trigger['stalta'].append(np.mean(trigger['cft_peaks']))

    print('Number of correlation triggered outliers:', len(corr_trigger['time']))

    df = pd.DataFrame(data={'duration': [trigger['duration'] for trigger in trig],
                            'stalta': [np.mean(trigger['cft_peaks']) for trigger in trig],
                            'time': [trigger['time'] for trigger in trig]})

    df2 = pd.DataFrame(data={'duration': corr_trigger['duration'],
                            'correlation': corr_trigger['correlation'],
                            'time': corr_trigger['time']})

    df.set_index('time', inplace=True)
    df.to_csv(f'tf/output/live/{station_org}_stalta_start_{start}.csv')
    df2.to_csv(f'tf/output/live/{station_org}_outliers_start_{start}.csv')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("station")
    parser.add_argument('start', nargs='?', default="2021-12-01")
    parser.add_argument('stop', nargs='?', default="2022-12-01")
    parser.add_argument('stalta_threshold', nargs='?', default=None)
    parser.add_argument('corr_threshold', nargs='?', default=None)

    args = parser.parse_args()

    # Make sure you use same parameters as for training!
    # input time window for training auto-encoder in seconds
    time_window_length = 20
    station = args.station
    if station == 'EKBG1':
        flow = 0.3
        fhigh = 24.9
    else :
        flow = 0.3
        fhigh = 12.49
    # processing data day-wise but longer/shorter possible
    duration = 3600*24
    start = UTCDateTime(args.start)
    stop = UTCDateTime(args.stop) 
    stalta_threshold = float(args.stalta_threshold)
    corr_threshold = float(args.corr_threshold)

    plot_level= False
    # Plot all STA/LTA detections
    #plot_level = 1
    # Plot all outlier detections
    #plot_level = 2

    modelpath = "/projects/active/MMON/GeobyIT/ML_DEVELOPEMENT/outliers/"
    modelpath = "./"
    print("Loading model...")
    timeseries_size = next_pow_2(2*time_window_length*fhigh)
    model = CAE((timeseries_size,3), base_depth=64, latent_factor=2)
    model.predict(np.random.random((8,timeseries_size,3)))
    model.load_weights(modelpath + f'tf/output/{station}_bn_elu.h5')
    print('Model loaded.')

    while start + duration <= stop :
        try :
            print("Processing day from",start)
            rshake = map_station_to_shake(station,start,start+duration)
            print(f'Station {station} is sensor {rshake}')
            main(rshake, station,start,duration,model,time_window_length,flow,fhigh,stalta_threshold,corr_threshold,plot_level)
        except Exception as e :
            print("Skipping time step",start,str(e)) 
            pass
        start += duration
