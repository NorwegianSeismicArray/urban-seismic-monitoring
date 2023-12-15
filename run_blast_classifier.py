# Copyright 2023 Andreas Koehler, MIT license

"""
Code for blast classification based on STA/LTA detections

"""

import tensorflow as tf
import pickle
#import pickle5 as pickle
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tqdm import tqdm
from obspy.core import UTCDateTime, Stream

from classifier import create_model
from utils import create_waveforms, map_station_to_shake

import numpy as np
import pandas as pd

from obspy.signal.trigger import coincidence_trigger

# Data is not on the Norwegian EIDA node yet
# Code will only run fro now if you adapt to your onw data
#from obspy.clients.fdsn import Client
#client = Client("UIB-NORSAR")
from obspy.clients.filesystem.sds import Client
sds_path = '/adhoc/seed/myshake/'
client = Client(sds_path)


def get_window_data(triggers,stream,winlen=80,flow=0.8,fhigh=25.5):

    windows = [(trig['time']-winlen/2., trig['time']+winlen/2.) for trig in triggers]
    tmp =[create_waveforms(es, ee, stations=None, stream=stream,
                           winlen=winlen, overlap=0, bandpass=(flow,fhigh)) 
                           for es, ee in tqdm(windows, desc="Loading data")]
    tmp = list(list(zip(*tmp))[0])
    X = np.concatenate(tmp, axis=0)
    return X,windows


def do_stalta(st,threshold = None, plot = False, thr_coincidence_sum = None, fmin=0.8, fmax=2.0):

    if threshold is None :
        threshold=7.0
    else :
        threshold=float(threshold)

    if thr_coincidence_sum is None :
        thr_coincidence_sum=3
    else :
        thr_coincidence_sum=float(thr_coincidence_sum)

    for tr in st:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled(0.0)
    st.detrend('demean')
    if fmin is not None:
        st.filter('highpass', freq=fmin)
    if fmax is not None:
        st.filter('lowpass', freq=fmax)

    trig = coincidence_trigger("recstalta", threshold, 1, st, thr_coincidence_sum, sta=0.5, lta=10, details = True)
    print('Number of triggers:', len(trig))
    if plot :
        for trigger in trig:
            print(trigger['time'],np.mean(trigger['cft_peaks']))
            st.select(channel='*Z*').slice(starttime=trigger['time']-30,endtime=trigger['time']+30).plot(equal_scale=False)
    return trig


def make_prediction(models,X):
    return np.mean([model.predict(X) for model in models],axis=0)

def main(modelpath,outpath,starttime,endtime,stations,flow,fhigh,fmin_trigger,fmax_trigger,crop,winlen,thrc,thrp) :

    eventpos = 0
    nsamples = winlen * 100

    print("loading models ...")
    with open(modelpath + f'/best_hps{suffix}.pkl', 'rb') as f:
        hps = pickle.load(f)
    #print(hps.values)
    fout=open(f'{outpath}results_statistics{suffix}.dat','w')

    models = []
    shape = (nsamples,len(stations)*3)
    for i in range(5):
        path_to_weights = modelpath + f'/models/blast_fold_{i}_weights{suffix}.h5'
        model = create_model(hps,2,crop=crop)
        model.predict(np.random.random((128,*shape)))
        model.load_weights(path_to_weights)
        #print(model.summary())
        models.append(model)

    print("... finished")

    stime = starttime
    blasttimes=[]
    blasttimes_end=[]
    p_max_all=[]
    all_finds = 0
    all_triggers = 0
    pstime = None

    # processing hour-wise
    while stime+3600 <= endtime :
        print(stime)
        if pstime is not None :
          if stime.julday != pstime.julday:
            data={'time': blasttimes, 'time_end' : blasttimes_end, 'score': p_max_all}
            df = pd.DataFrame(data)
            date = str(pstime)[:10]
            df.to_csv(f'{outpath}blast_detections_{date}{suffix}.csv', float_format='%.1f')
            fout.write("Number of triggers %s: %d\n" % (str(pstime)[:10],all_triggers))
            fout.write("Number of blasts %s: %d\n" % (str(pstime)[:10],all_finds))
            fout.flush()
            blasttimes=[]
            blasttimes_end=[]
            p_max_all=[]
            all_finds = 0
            all_triggers = 0

        st = Stream()
        for stat in stations:
            try :
                st = st + client.get_waveforms('*',map_station_to_shake(stat,stime-120,stime+3600+120),'*','*', stime-120, stime+3600+120)
            except Exception :
                break
        st.merge()
        if len(set([tr.stats.station for tr in st])) != len(stations) :
            print(set([tr.stats.station for tr in st]))
            print("Not all stations found for", stime)
            pstime = stime
            stime += 3600.
            continue

        triggers = do_stalta(st.copy().slice(starttime=stime,endtime=stime+3600),threshold = None,
                             plot = False, thr_coincidence_sum = thrc,fmin=fmin_trigger,fmax=fmax_trigger)
        if len(triggers) == 0 :
            pstime = stime
            stime += 3600.
            continue
        all_triggers += len(triggers)

        #try :
        if True:
            X,windows = get_window_data(triggers,st,winlen,flow=flow,fhigh=fhigh)
            X = TimeSeriesScalerMeanVariance().fit_transform(X)
            predictions = make_prediction(models,X)
        #except ValueError :
        #    print("Something went wrong for",stime)
        #    pstime = stime
        #    stime += 3600.
        #    continue
        print("Prediction finished for",stime)
        i = 0
        for t,p in zip(windows,predictions):
            output="%s %1.2f\n" % ( str(t[0]+(t[1]-t[0])/2),p[eventpos])
            print(output)
            if p[eventpos] > thrp :
                print("FOUND BLAST")
                blasttimes.append(str(t[0])[:23])
                blasttimes_end.append(str(t[1])[:23])
                p_max_all.append(p[eventpos])
                all_finds += 1
                i=i+1
        pstime = stime
        stime += 3600.
    # write last day
    data={'time': blasttimes, 'time_end' : blasttimes_end, 'score': p_max_all}
    df = pd.DataFrame(data)
    date = str(pstime)[:10]
    df.to_csv(f'{outpath}blast_detections_{date}{suffix}.csv', float_format='%.1f')
    fout.write("Number of triggers %s: %d\n" % (str(pstime)[:10],all_triggers))
    fout.write("Number of blasts %s: %d\n" % (str(pstime)[:10],all_finds))
    fout.close()

if __name__ == '__main__':

    modelpath = "tf/output"
    outpath = 'tf/output/live/'

    starttime = UTCDateTime("2022-06-01T00:00:00")
    endtime = UTCDateTime("2023-10-01T00:00:00")
    starttime = UTCDateTime("2022-10-10T00:00:00")
    endtime = UTCDateTime("2022-10-11T00:00:00")


    stations = ['OSLN2']

    # Must be the same as for training data
    flow = 0.8
    fhigh = 25.5
    crop = True
    if crop :
        suffix = '_crop_True'
        winlen = 26
    else :
        crop = False
        suffix = '_crop_False'
        winlen = 22

    # STA/LTA trigger
    # Trigger on RGs
    fmin_trigger = 0.8
    fmax_trigger = 2.0
    # For single station classifier only one channel has to trigger
    thrc  = 1
  
    # Classification threshold
    thrp = 0.7

    main(modelpath,outpath,starttime,endtime,stations,flow,fhigh,fmin_trigger,fmax_trigger,crop,winlen,thrc,thrp)
    print("Finished")
