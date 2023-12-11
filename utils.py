# Copyright 2023 Andreas Koehler, MIT license

from obspy import UTCDateTime
from obspy.core import Stream
import numpy as np
from datetime import datetime

# Data is not on the Norwegian EIDA node yet
# Code will only run fro now if you adapt to your onw data
#from obspy.clients.fdsn import Client
#client = Client("UIB-NORSAR")
from obspy.clients.filesystem.sds import Client
sds_path = '/adhoc/seed/myshake/'
client = Client(sds_path)

def create_windows(start,end,winlen,overlap):
    """
    Create time windows from start to end with lenght winlen and overlap between windows. 
    """

    assert type(start) == UTCDateTime
    assert type(end) == UTCDateTime
    windows = []
    curr_time = start
    while curr_time < end:
        x = []
        old_time = curr_time - overlap
        curr_time += winlen - overlap
        windows.append((old_time,curr_time))
    return windows

def create_waveforms(start, end, stations, winlen=60, overlap=30, normalize=False, bandpass=(1,49.9), downsample_factor=1):
    """
    start : UTCDateTime
    end : UTCDateTime
    stations : list of str, names of stations.
    winlen : int, waveform window size in seconds. 
    overlap : int, overlap between windows. < winlen
    """
    assert start < end
    assert overlap < winlen
    assert type(start) == UTCDateTime
    assert type(end) == UTCDateTime

    fmin, fmax = bandpass

    start -= winlen
    end += winlen
    st = Stream()
    for stat in stations:
        st = st + client.get_waveforms('*',stat,'*','*', start, end)
    st.merge()

    i = -1
    for i,tr in enumerate(st):
        df = tr.stats.sampling_rate
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled(0.0)

    if i < 0: return np.asarray([]), np.asarray([])

    st.detrend('demean')
    st.taper(max_percentage=0.05)
    if fmin is not None:
        st.filter('highpass', freq=fmin)
    if fmax is not None:
        st.filter('lowpass', freq=fmax)

    if downsample_factor > 1:
        df = df // downsample_factor
        st.resample(df, strict_length=False)

    st.trim(start+1, end-1)
    
    x = []
    for tr in st:
        #print(tr.stats.station,tr.stats.channel)
        tmp = tr.data.reshape((-1,1))
        time = tr.times("utcdatetime")
        if normalize:
            tmp = (tmp - tmp.mean())/tmp.std()
        x.append(tmp)

    x = np.squeeze(np.asarray(x)).T

    winlen = int(df * winlen)
    overlap = int(df * overlap)
    
    out = []
    windows = []
    for i in range(winlen, x.shape[0]-winlen, winlen-overlap):
        w = x[i:i+winlen]
        t = time[i:i+winlen]
        if len(w) == winlen:
            windows.append((t[0],t[-1]))
            out.append(w)

    out = np.asarray(out)
    windows = np.asarray(windows)

    assert np.isreal(out).all()

    return out, windows , df

def map_station_to_shake(station,starttime,endtime):
   statdict ={
   'ALNN1' : 'R242B',
   'ALNN2' : 'R88BC',
   'ALNN3' : 'R0A6A',
   'ALNN4' : 'R1E14',
   'ALNN5' : 'RC12E',
   'ALNN6' : 'R81A2',
   'ALNN7' : 'R7C69',
   'ALNN8' : 'R43C9',
   'EKBG1' : 'R17B2',
   'OSLN1' : 'R1E14',
   'OSLN2' : 'R7772',
   'OSLN3' : 'RA102',
   'OSLN4' : 'RD570',
   'OSLN5' : 'RD1E4'
   }

   if station == 'ALNN1' or station == 'ALNN2':
       if starttime < UTCDateTime("2021-06-08T10:00:00") and endtime < UTCDateTime("2021-06-08T10:00:00") :
           statdict['ALNN1']='R591E'
           statdict['ALNN2']='R17B2'
       if starttime < UTCDateTime("2021-06-08T10:00:00") and endtime > UTCDateTime("2021-06-08T10:00:00") :
           raise Exception("Instrument changed during processed time window. Skipping.")
   if station == 'ALNN4' :
       if starttime > UTCDateTime("2022-04-08T08:00:00"):
           statdict['ALNN4']='RD570'
       if starttime < UTCDateTime("2022-04-08T08:00:00") and endtime > UTCDateTime("2022-04-08T08:00:00") :
           raise Exception("Instrument changed during processed time window. Skipping.")
       if starttime > UTCDateTime("2022-09-30T11:00:00"):
           raise Exception("Instrument running at another location. Skipping.")
   if station == 'ALNN8' :
       if starttime < UTCDateTime("2021-11-15T09:53:00"):
           statdict['ALNN8']='R591E'
       if starttime < UTCDateTime("2021-11-15T09:53:00") and endtime > UTCDateTime("2021-11-15T09:53:00") :
           raise Exception("Instrument changed during processed time window. Skipping.")
   if station == 'EKBG1' :
       if starttime < UTCDateTime("2022-02-11T10:00:00"):
           statdict['EKBG1']='RD570'
       if starttime < UTCDateTime("2022-02-11T10:00:00") and endtime > UTCDateTime("2022-02-11T10:00:00") :
            raise Exception("Instrument changed during processed time window. Skipping.")
   if station == 'ALNN2' and starttime > UTCDateTime("2021-11-15T10:40:00"):
       raise Exception("Instrument moved to Kjeller during processed time window. Skipping.")
   if station == 'ALNN7' and starttime < UTCDateTime("2021-10-11T09:00:00"):
       raise Exception("Instrument running in Kjeller during processed time window. Skipping.")
   if station == 'OSLN1' and starttime < UTCDateTime("2022-04-26T15:00:00"):
       raise Exception("Instrument running at another location. Skipping.")
   if station == 'OSLN2' and starttime < UTCDateTime("2022-06-01T08:35:00"):
       raise Exception("Instrument running at another location. Skipping.")
   if station == 'OSLN3' and starttime < UTCDateTime("2022-06-01T09:45:00"):
       raise Exception("Instrument running at another location. Skipping.")
   if station == 'OSLN4' and starttime < UTCDateTime("2022-10-24T10:00:00"):
       raise Exception("Instrument running at another location. Skipping.")
   if station == 'OSLN5' and starttime < UTCDateTime("2023-06-09T13:00:00"):
       raise Exception("Instrument running at another location. Skipping.")

   return statdict[station]

def recording_periods():
    deployment = {}
    deployment['EKBG1'] = (UTCDateTime("2021-11-03T00:00:01"), None)
    deployment['ALNN1'] = (UTCDateTime("2021-05-21T00:00:01"), None)
    deployment['ALNN2'] = (UTCDateTime("2021-05-21T00:00:01"), UTCDateTime("2021-11-15T00:00:01"))
    deployment['ALNN3'] = (UTCDateTime("2021-05-21T00:00:01"), None)
    deployment['ALNN7'] = (UTCDateTime("2021-10-12T00:00:01"), None)
    deployment['ALNN4'] = (UTCDateTime("2021-06-09T00:00:01"), UTCDateTime("2022-09-30T00:00:01"))
    deployment['ALNN5'] = (UTCDateTime("2021-09-26T00:00:01"), None)
    deployment['ALNN6'] = (UTCDateTime("2021-09-30T00:00:01"), None)
    deployment['ALNN8'] = (UTCDateTime("2021-11-16T00:00:01"), None)
    deployment['OSLN1'] = (UTCDateTime("2022-04-26T14:30:01"), None)
    deployment['OSLN2'] = (UTCDateTime("2022-06-01T12:00:01"), None)
    deployment['OSLN3'] = (UTCDateTime("2022-06-01T12:00:01"), None)
    deployment['OSLN4'] = (UTCDateTime("2022-10-24T10:00:01"), UTCDateTime("2023-06-24T00:00:01"))
    deployment['OSLN5'] = (UTCDateTime("2023-06-09T13:00:01"), None)
    return deployment
