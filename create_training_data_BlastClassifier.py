# Copyright 2023 Andreas Koehler, MIT license

"""
Code for generating input data for blast classification

"""

import numpy as np
from utils import create_waveforms, map_station_to_shake
from obspy.core import UTCDateTime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def load_events(filename):
    events = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            events.append(UTCDateTime(l.strip()))
    return events

if __name__ == '__main__':

    # single station classifier using OSLN2 only
    # Add more stations to train a classifier with samples from different stations
    stationlist = ['OSLN2']
    # Files with blast detection times for training
    event_files = ['/tf/data/blast-signals-OSLN2.txt']
    # Additional noise class examples (noise window before each event is always chosen)
    noise_files = ['/tf/data/noise.txt']

    # Random cropping of input data
    crop = True
    if crop : increased_size = 0.25

    # Root directory for tensorflow input and output
    datapath ='.'

    # Bandpass filter for input waveforms
    flow = 0.8
    fhigh = 25.5
    
    # Time window duration in seconds. Will be increased if crop is True
    winlen = 22
    if crop:
        offset = increased_size*winlen/2
    else:
        offset = 0
    offset = int(offset)

    outX = []
    outy = []
    outE = []

    print("Preparing data ...")

    for i, fname in enumerate([datapath+event_file for event_file in event_files]):
        events = load_events(fname)
        events = [(e-offset-winlen//2, e+winlen//2+offset) for e in events]
        # For noise class take time window before blast: 2x winlen time difference
        events2 = load_events(fname)
        events2 = [(e-offset-winlen//2-2*winlen, e+winlen//2+offset-2*winlen) for e in events2]
        for station in stationlist :
            blasts_and_win = [create_waveforms(es, ee, [map_station_to_shake(station,es,ee)],
                             winlen=winlen + 2*offset, bandpass=(flow,fhigh),
                             overlap=0) for es, ee in tqdm(events, desc="Loading blasts")]
            blasts = list(list(zip(*blasts_and_win))[0])
            eventwindows = list(list(zip(*blasts_and_win))[1])
            blasts = np.concatenate(blasts, axis=0)
            eventwindows = np.concatenate(eventwindows, axis=0)

            noise_and_win = [create_waveforms(sn, en, [map_station_to_shake(station,sn,en)],
                             winlen=winlen + 2*offset, bandpass=(flow,fhigh),
                             overlap=0) for sn, en in tqdm(events2, desc="Loading noise")]
            noise = list(list(zip(*noise_and_win))[0])
            noisewindows = list(list(zip(*noise_and_win))[1])
            noise = np.concatenate(noise, axis=0)
            noisewindows = np.concatenate(noisewindows, axis=0)

            events = np.concatenate([eventwindows, noisewindows], axis=0)
            X = np.concatenate([blasts, noise], axis=0)
            y = ['blast'] * len(blasts) + ['noise'] * len(noise)
            if len(X) != len(y) :
                print("Number of data and labels does not match!")
                exit()
            print("Added data samples (blasts and noise):",len(X))
            outX.append(X)
            outy.append(y)
            outE.append(events)

    # add additional examples of noise
    for i, fname in enumerate([datapath+noise_file for noise_file in noise_files]):
        events = load_events(fname)
        events = [(e-offset-winlen//2, e+winlen//2+offset) for e in events]
        for station in stationlist :
            noise_and_win = [create_waveforms(sn, en, [map_station_to_shake(station,sn,en)],
                             winlen=winlen + 2*offset, bandpass=(flow,fhigh),
                             overlap=0) for sn, en in tqdm(events, desc="Loading noise")]
            noise = list(list(zip(*noise_and_win))[0])
            noisewindows = list(list(zip(*noise_and_win))[1])
            noise = np.concatenate(noise, axis=0)
            noisewindows = np.concatenate(noisewindows, axis=0)

            X = noise
            y = ['noise'] * len(noise)
            print("Addded data samples (noise):",len(X))
            outX.append(X)
            outy.append(y)
            outE.append(events)

    X = np.concatenate(outX, axis=0)
    y = np.concatenate(outy, axis=0)
    y = LabelEncoder().fit_transform(y)
    events = np.concatenate(outE, axis=0)

    np.save(f'{datapath}/tf/data/blastclassifier_data_crop_{crop}.npy', X)
    np.save(f'{datapath}/tf/data/blastclassifier_labels_crop_{crop}.npy', y)
    np.save(f'{datapath}/tf/data/blastclassifier_timewindows_crop_{crop}.npy', events)
