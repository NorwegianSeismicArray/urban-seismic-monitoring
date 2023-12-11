# Copyright 2023 Andreas Koehler, MIT license

"""
Code for generating input data for auto-encoder training

"""

from utils import create_waveforms, create_windows, map_station_to_shake, recording_periods
import numpy as np
from obspy.core import UTCDateTime
from obspy.signal.util import next_pow_2
from tslearn.preprocessing import TimeSeriesResampler
from tqdm import tqdm

deployment = recording_periods()

def load_data(filename, station,
              train_size=12 * 3600,
              winlen=60,
              overlap=0,
              filter=None):

    time_periods = []
    try : deployment_this_station, end_this_station = deployment[station]
    except KeyError :
        print(f'{station} not found')
        exit()
    if end_this_station is None:
        end_this_station = UTCDateTime.now()

    if type(filename) == str:
        with open(filename, 'r') as f:
            for l in f.readlines():
                start = UTCDateTime(l.strip().split()[0])
                if start > deployment_this_station and start < end_this_station :
                    time_periods.append(dict(start=start, station=station))
                else:
                    print(f'{station} not producing data for time period at {start}.')
        time_periods = list(sorted(time_periods, key=lambda item: item['start']))
    else:
        raise NotImplementedError(filename + ' not supported.')

    def create_train_data(time_period):
        start = max(time_period['start'], deployment_this_station)
        end = min(time_period['start'] + train_size, end_this_station)

        stations = list(deployment.keys()) if time_period['station'] == 'ALL' else [time_period['station']]
        try :
            stations = [map_station_to_shake(sta,start,end) for sta in stations]
            results = create_waveforms(start, end, stations,
                         winlen=winlen,
                         overlap=overlap,
                         bandpass=filter,
                         downsample_factor=1)
        except Exception as e:
            print(e)
            results = None
        return results

    print("creating training data")
    train = map(create_train_data, time_periods)
    train = [(a,b,c) for a,b,c in tqdm(train, total=len(time_periods)) if a.size > 0]
    train, window_train, df = list(zip(*train))
    return np.concatenate(train, axis=0), \
           np.concatenate(window_train, axis=0), df[0]

if __name__ == '__main__':

    # seismic station
    station = 'EKBG1'
    station = 'OSLN2'
    # input time window for training auto-encoder in seconds
    time_window_length = 20 
    # each line contains a start time to define training data
    noisefile = 'training_windows_'+station+'.txt'
    # record duration for training starting from time in noisefile
    train_size = 12*3600
    train_size = 3600
    # time window overlap in seconds
    overlap = 5
    # frequency band
    if station == 'EKBG1':
        flow = 0.3
        fhigh = 24.9
    else :
        flow = 0.3
        fhigh = 12.49
    rootpath = './'

    print('Loading data from station: ', station)
    train, train_w, df = load_data(rootpath+noisefile,
                                             station,
                                             winlen=time_window_length,
                                             overlap=overlap,
                                             train_size=train_size,
                                             filter=(flow,fhigh))

    # Sub-sample the time window to save training time:
    # Sub-sampling is done with closest power of two duration
    # E.g., for a station with 100Hz sampling and for fmax=25Hz
    # we only need 50 Hz sampling. With 20s window length this corresponds to
    # 1000s samples -> Number of time samples in auto-encoder is 1024
    timeseries_size = next_pow_2(2*time_window_length*fhigh)
    print("Final number of time samples for training:",timeseries_size)
    train = TimeSeriesResampler(timeseries_size).fit_transform(train)

    np.save(f'{rootpath}tf/data/train_{station}.npy', train)
    np.save(f'{rootpath}tf/data/windows_train_{station}.npy', train_w)
