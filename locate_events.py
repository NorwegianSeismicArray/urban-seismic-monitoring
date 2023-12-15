# Copyright 2023 Andreas Koehler, MIT license

"""
Code for locating detected events

"""

from obspy import UTCDateTime
import numpy as np
import pandas as pd
import argparse
import glob

# Data is not on the Norwegian EIDA node yet
# Code will only run for now if you adapt to your onw data
from obspy.clients.fdsn import Client
client_ndb = Client("UIB-NORSAR")
from obspy.clients.filesystem.sds import Client
client = Client('/adhoc/seed/myshake/')

from utils import create_oslo_RgTTtabel, get_stations
from utils import get_waveforms, preprocess_waveforms, get_baz, get_loc_stackmap, plot_results

parser = argparse.ArgumentParser()
# Trigger / classifier station
parser.add_argument("station", default='OSLN2')
# Start time
parser.add_argument('start', nargs='?', default="1970-01-01")
# "ouliers" : outlier evnets
# "stalta" : sta lta triggers
# "classifier" : events classified as blasts by CNN
parser.add_argument('mode', nargs='?', default="outliers")
args = parser.parse_args()

if __name__ == '__main__':

    ############################### define input parameters ###################################

    # where csv files of outlier detection are stored
    outlierpath = "tf/output/live/"
    if args.mode == 'classfier' :
        outlierpath = "tf/output/blasts"
  
    # Plot waveforms and envelopes for interactive processing
    plot = False
    plot = True
    # Use all three components for stacking and location
    all_comp = True
    all_comp = False

    # First station is trigger station - all stations are used for location
    trigger_stat = args.station
    stat = 'OSLN2,OSLN3,EKBG1,OSLN1,OSLN4,OSLN5,ALNN1,ALNN3,ALNN4,ALNN5,ALNN6,ALNN7'
    if trigger_stat == 'EKBG1':
        stat = 'EKBG1,OSLN2,OSLN3,OSLN1,OSLN4,OSLN5,ALNN1,ALNN3,ALNN4,ALNN5,ALNN6,ALNN7'

    # Frequency band for plotting waveforms
    fmin = 0.8
    fmax = 25.5
    # Frequency bands for locating event with RG wave
    rgfmin = 0.8
    rgfmax = 2.0

    # Signal-to-Noise ratio threshold for using back-azimuth of an event
    snrthr=7.0
    if trigger_stat == 'EKBG1' :  snrthr=6.1

    # Minimum number of stations/channels for event location (stacking)
    minstat = 3
    if  all_comp : minstat = 9

    # Weight factor when substracting back-azimuth residual from stacked envelopes during location
    bazw = 1./1000
    
    # Bounds for location grid search (lon_min, lon_max, lat_min, lat_max):
    extent = [10.55,11.00, 59.86,60]
    # Step for grid serach
    dlon = 0.01
    dlat = 0.005
    # RG velocity in m/s
    rg_vel = 2000

    # Time window in seconds relative to event detection time
    tw_start = -10
    tw_end = 35

    #########################################################################################################

    # Create travel time table
    tt_tab,locs,nlon,nlat=create_oslo_RgTTtabel(extent[0],extent[1],extent[2],extent[3],dlon,dlat,rg_vel)
    statlon,statlat=get_stations()
    statlon=[statlon[key] for key in statlon]
    statlat=[statlat[key] for key in statlat]

    if args.mode == 'stalta' : fout= open(outlierpath+'located_events_stalta.out','w')
    elif args.mode == 'classifier' : fout= open(outlierpath+'located_events_classifier.out','w')
    else : fout= open(outlierpath+'located_events.out','w')

    print("Reading detections ...")
    eventlist = []
    if args.mode != 'classifier' :
        duration = []
        stalta_vals = []
        for outlierfile in sorted(glob.glob(outlierpath+trigger_stat+"_"+args.mode+"*.csv")) :
            date = outlierfile[-14:-4]
            df = pd.read_csv(outlierfile)
            idx=0
            for event,dur in zip(df['time'].to_list(),df['duration'].to_list()) :
                eventlist.append(UTCDateTime(event))
                duration.append(dur)
                if args.mode == 'stalta': stalta_vals.append(df['stalta'].to_list()[idx])
                idx += 1
    else :
        for outlierfile in sorted(glob.glob(outlierpath+"blast_detections_*__stalta_"+args.mode+".csv")) :
            df = pd.read_csv(outlierfile)
            for event,end in zip(df['time'].to_list(),df['time_end'].to_list()) :
                eventlist.append(UTCDateTime(event)+(UTCDateTime(end)-UTCDateTime(event))/2.)

    for idx,event in enumerate(eventlist):
        print("Processing detection:",event)
        if event > UTCDateTime(args.start):
            st = get_waveforms(event,stat,tw_start,tw_end,client,client_ndb)
            if st == None : continue
            st_env,st_3c_rgfilt,st_3c_filt,times,snr,snr_comp = preprocess_waveforms(
                                                  event,st,fmin,fmax,rgfmin,rgfmax,allcomp=all_comp)

            baz = get_baz(st_env,st_3c_rgfilt,snr,snrthr,times)
            if len(baz)>0 :
                print("Back-azimuth measurements:")
                print(baz)
            else : print("No back-azimuth measurements for this event.")

            # grid search and envelope stacking
            if all_comp : snr = snr_comp
            res,stop  = get_loc_stackmap(event,st_env,locs,tt_tab,snr,snrthr,baz,bazw,minstat)
            if stop :
                print("This detection is not locatable.")
                continue

            if plot : plot_results(event,st_env,st_3c_filt,locs,res,extent,statlon,statlat)

            include_for_training=input("Give an event label or skip event (no input):")
            if include_for_training :
                if args.mode != 'classifier' : dur=duration[idx]
                else : dur=0
                fout.write("%s %f %f %f %s\n" % (str(event),locs[np.argmax(res)][0],locs[np.argmax(res)][1],dur,include_for_training))
                fout.flush()
    fout.close()
