# Copyright 2023 Andreas Koehler, MIT license

from obspy import UTCDateTime
from obspy.core import Stream
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.filter import envelope
from scipy.signal import hilbert
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import cartopy.mpl as mpl


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

def get_stations():
    stationlon={
    'EKBG1': 10.75811,
    'OSLN1': 10.76942,
    'OSLN2': 10.70622,
    'OSLN3': 10.73275,
    'OSLN4': 10.65481,
    'OSLN5': 10.76700,
    'OSL':  10.72169,
    'OFSN2': 10.9108,
    'ALNN1': 10.85820,
    'ALNN3': 10.84520,
    'ALNN4': 10.84797,
    'ALNN5': 10.83357,
    'ALNN6': 10.83525,
    'ALNN7': 10.84636,
    'ALNN8': 10.83729
    }

    stationlat={
    'EKBG1': 59.89741,
    'OSLN1': 59.95523,
    'OSLN2': 59.94153,
    'OSLN3': 59.94254,
    'OSLN4': 59.94149,
    'OSLN5': 59.96505,
    'OSL': 59.93734,
    'OFSN2': 59.8401,
    'ALNN1': 59.93358,
    'ALNN3': 59.92998,
    'ALNN4': 59.93144,
    'ALNN5': 59.94091,
    'ALNN6': 59.94045,
    'ALNN7': 59.93023,
    'ALNN8': 59.94112
    }
    return stationlon,stationlat


def create_oslo_RgTTtabel(lonmin,lonmax,latmin,latmax,dlon,dlat,rg_vel=2500):

    stationlon,stationlat=get_stations()

    tt_table={
    'EKBG1': {'TT': [],'BAZ': []},
    'OSLN1': {'TT': [],'BAZ': []},
    'OSLN2': {'TT': [],'BAZ': []},
    'OSLN3': {'TT': [],'BAZ': []},
    'OSLN4': {'TT': [],'BAZ': []},
    'OSLN5': {'TT': [],'BAZ': []},
    'OSL': {'TT': [],'BAZ': []},
    'OFSN2': {'TT': [],'BAZ': []},
    'ALNN1': {'TT': [],'BAZ': []},
    'ALNN3': {'TT': [],'BAZ': []},
    'ALNN4': {'TT': [],'BAZ': []},
    'ALNN5': {'TT': [],'BAZ': []},
    'ALNN6': {'TT': [],'BAZ': []},
    'ALNN7': {'TT': [],'BAZ': []},
    'ALNN8': {'TT': [],'BAZ': []}
    }

    locations=[]

    for lon in np.arange(lonmin,lonmax,dlon):
        for lat in np.arange(latmin,latmax,dlat):
            locations.append([lon,lat])
            for stat in tt_table:
                dist,_,baz=gps2dist_azimuth(lat, lon, stationlat[stat], stationlon[stat])
                tt_table[stat]['TT'].append(dist/rg_vel)
                tt_table[stat]['BAZ'].append(baz)
    return(tt_table,locations,len(np.arange(lonmin,lonmax,dlon)),len(np.arange(latmin,latmax,dlat)))

def get_waveforms(event,stat,tw_start,tw_end,client,client_ndb):
    if ',' in stat :
        st=Stream()
        for stat2 in stat.split(','):
            try :
                stt = client.get_waveforms('*',map_station_to_shake(stat2,event+tw_start, event+tw_end),'*','*', event+tw_start, event+tw_end)
            except :
                continue
            for tr in stt : tr.stats.station = stat2
            st += stt
    else :
        try :
            st = client.get_waveforms('*',map_station_to_shake(stat,event+tw_start, event+tw_end),'*','*', event+tw_start, event+tw_end)
        except :
            return None
    try :
        st += client_ndb.get_waveforms('*','OSL','*','*', event+tw_start, event+tw_end)
    except OSError or _pickle.UnpicklingError :
        pass
    return st

def preprocess_waveforms(event,st,fmin,fmax,rgfmin,rgfmax,allcomp=False):
    for tr in st:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled(0.0)
    st.detrend('demean')
    st.taper(max_percentage=0.05)
    st_3c_filt = st.copy().filter('bandpass', freqmin=fmin,freqmax=fmax)
    st_3c_rgfilt = st.copy().filter('bandpass', freqmin=rgfmin,freqmax=rgfmax)
    st_3c_rgfilt.trim(event-10,event+20)
    st_3c_filt.trim(event-10,event+20)

    # channels used for envelope stacking
    if not allcomp:
        st=st.select(channel="*Z*")
    st.filter('bandpass', freqmin=rgfmin,freqmax=rgfmax,zerophase=True)
    st_3c_rgfilt.trim(event-10,event+20)
    st_3c_filt.trim(event-10,event+20)
    times={}
    snr={}
    snr_comp={}
    # compute snr and time of envelope maximum
    for tr in st:
        if tr.stats.sampling_rate != 100.0 : tr.resample(100)
        tr.data = envelope(tr.data)
        tr.data /= tr.data.max()
        try :
            times[tr.stats.station]=np.argmax(tr.data[:-1500])/tr.stats.sampling_rate
            if not allcomp: snr[tr.stats.station]=np.max(tr.data[:-1500])/np.mean(tr.data[100:500])
            else : snr_comp[tr.stats.station+'-'+tr.stats.channel]=np.max(tr.data[:-1500])/np.mean(tr.data[100:500])
        except ValueError :
            continue
    return st,st_3c_rgfilt,st_3c_filt,times,snr,snr_comp

def get_baz(st,st_3c,snr,snrthr,times):
    baz={}
    for stat1 in set([tr.stats.station for tr in st]):
        try :
            test=snr[stat1]
        except KeyError :
            continue
        if snr[stat1]>snrthr:
            st_rot=st_3c.copy().select(station=stat1)
            if stat1 == 'OSLN3': correct=90.
            else : correct=None
            try :
                angle,zr_corr,_,_,_=find_radial(st_rot,st_rot[0].stats.starttime+times[stat1],True,350,350,False,correct)
                if zr_corr > 0.7 : baz[stat1]=angle
            except ValueError :
                continue
    return baz

def get_loc_stackmap(event,st,locs,tt_tab,snr,snrthr,baz,bazw,minstat=4,plot=True,use_baz=True):
    st_cp = st.copy()
    for tr in st_cp:
        try :
            if snr[tr.stats.station]<=snrthr :
                st_cp.remove(tr)
        except KeyError :
            try :
                if snr[tr.stats.station+'-'+tr.stats.channel]<=snrthr :
                    st_cp.remove(tr)
            except KeyError :
                continue
    res=np.array([0.0]*len(locs))
    stop = False
    maxres = 0.
    besti = 0
    for i,tt in enumerate(tt_tab[st[0].stats.station]['TT']):
        st_stack = st_cp.copy()
        for tr in st_stack:
            tr.stats.starttime -= tt_tab[tr.stats.station]['TT'][i]
        if len(st_stack) >= minstat :
            st_stack.trim(event-10,event+20)
            try :
                res[i] = np.max(st_stack.stack(npts_tol=1)[0].data)
                if tr.stats.station in baz and use_baz:
                    res[i] -= bazw * np.abs(tt_tab[tr.stats.station]['BAZ'][i]-baz[tr.stats.station])
                if res[i] > maxres :
                    maxres = res[i]
                    besti = i
            except ValueError: continue
        else :
            stop = True
            break
    if plot and not stop :
        st_stack = st_cp.copy()
        for tr in st_stack:
            tr.stats.starttime -= tt_tab[tr.stats.station]['TT'][besti]
        print("Plot: Shifted envelopes of best location")
        st_stack.plot()
    return res,stop

def plot_results(event,st,st_3c_filt,locs,res,extent,statlon,statlat):
    request = cimgt.OSM()
    st.trim(event-10,event+20)
    st_3c_filt = st_3c_filt.trim(event-10,event+20).select(channel="*Z*")
    stz = st.copy().select(channel="*Z*")

    print("Plot: Waveform data")
    st_3c_filt.plot(equal_scale=False)
    stationlon,stationlat=get_stations()

    stz.trim(event-5,event+12.5)
    st_3c_filt.trim(event-5,event+12.5)
    plt.figure(figsize=(5,5))
    stats = list(set([tr.stats.station for tr in stz]))
    stats.sort()
    for tr in st_3c_filt : tr.data = tr.data / tr.data.max()
    # order stations by distance
    dists=[]
    for stat in stats:
        dist,_,baz=gps2dist_azimuth(stationlat[stat],stationlon[stat],locs[np.argmax(res)][1], locs[np.argmax(res)][0])
        dists.append(dist)
    idx=np.argsort(np.array(dists))
    for i,stat in enumerate(np.flip(np.array(stats)[idx])):
        times=st_3c_filt.select(station=stat)[0].times()
        plt.plot(times,st_3c_filt.select(station=stat)[0].data+i*2,c='black',linewidth=1)
        times=stz.select(station=stat)[0].times()
        plt.plot(times,stz.select(station=stat)[0].data+i*2,c='orange',linewidth=2)
    plt.yticks(np.array(range(len(stats)))*2, np.flip(np.array(stats)[idx]))
    plt.xlabel("Time (s)")
    plt.title(str(event)[:19])
    print("Plot: Waveforms and envelopes")
    plt.show()

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=request.crs)
    ax.set_extent([extent[0]-0.005,extent[1]-0.005,extent[2]-0.0025,extent[3]+0.0025])
    ax.add_image(request, 12)    # zoom level
    plt.scatter(np.transpose(locs)[0],np.transpose(locs)[1], transform=ccrs.PlateCarree(),c=res,alpha=0.4,marker='s',s=150,edgecolors = 'none')
    plt.colorbar(fraction=0.03, pad=0.1)
    plt.scatter(statlon,statlat,transform=ccrs.PlateCarree(),s=200, marker='^',edgecolors = 'k')
    ax.scatter(locs[np.argmax(res)][0], locs[np.argmax(res)][1], transform=ccrs.PlateCarree(),s=200,edgecolors = 'k')
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='none',lw=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = mpl.gridliner.LATITUDE_FORMATTER
    print("Plot: Location map")
    plt.show()

def rotate(st,angle,correct=None):
    sto=Stream()
    trz=st.copy().select(channel="*Z")
    trn=st.copy().select(channel="*N")
    tre=st.copy().select(channel="*E")
    # trim to same length
    if len(trn[0].data) != len(tre[0].data) :
        newstart=np.max([trn[0].stats.starttime,tre[0].stats.starttime])
        newend=np.min([trn[0].stats.endtime,tre[0].stats.endtime])
        trn=trn.trim(newstart,newend)
        tre=tre.trim(newstart,newend)
    # in rotate_ne_rt rotation angle=baz to get radial component for signal coming from baz
    # rotate data to North (not rotate North to data!)
    # no rotation corresponds to baz=180 -> R and N are identical (pos to North)
    # rotation with baz=0: flip of polarity for N to get R (in propagation direction)
    # rotation baz=10: flip and then rotation anti-clockwise 10 degrees
    # known misorientation of a sensor must be given as value clockwise from North
    # -> flip and then rotate anti-clockwise (*-1) to get actual signal from North
    # on the wrongly orientated North component
    if correct is not None : angle=angle-correct
    if angle > 360 : angle=angle-360
    if angle < 0 : angle=angle+360
    trn[0].data,tre[0].data=rotate_ne_rt(trn[0].data, tre[0].data, angle)
    sto=trz+trn+tre
    return sto


def find_radial(st,onset,do_hilbert,win1,win2,plotrot=False,correct=None) :
    start=int((onset-st[0].stats.starttime)*st[0].stats.sampling_rate)
    angrange=np.arange(0,360,1)
    # zr : correlation coefficient between vertical and radial component for all angles
    zr=[]
    # env : envelope of the radial component for all angles
    env=[]
    for angle in angrange :
        stz=st.copy().select(channel="*Z")
        stm=rotate(st.copy(),angle,correct)
        # select radial component which was rotated North component (is Radial compoment now)
        stm=stm.select(channel="*N")
        tm=envelope(stm[0].data)
        env.append(tm[start-win1:start+win2])
        if do_hilbert : stm[0].data=np.imag(hilbert(stm[0].data))
        zr.append(pearsonr(stz[0].data[start-win1:start+win2],stm[0].data[start-win1:start+win2])[0])
    # ind : index if maximum envelope amplitude for each angle
    ind=[np.argmax(a) for a in env]
    # val : maximum envelope amplitude for each angle
    val=[np.max(a) for a in env]
    # pick the angle that maximizes amplitude in correct range (180 ambiguity) so that R trace has same polarity as Z
    # a bit cryptic because all in one line, let's explain it:
    # Find 180 degree angle interval around maximum correlation (+/-90)
    # angle index of lower limit : np.argmax(zr)-int(len(val)/4)
    # angle index of upper limit : np.argmax(zr)+int(len(val)/4) 
    # angle index if max envelope in range, correcting for negative indices : np.argmax([val[i%len(val)] for i in ... ])
    # add lower limit index (corrected) : (np.argmax(zr)-int(len(val)/4))%len(val) + ...
    # correct again in the end to avoid neg. or angle index >len(val) : (...)%len(val)
    bestind=((np.argmax(zr)-int(len(val)/4))%len(val) + np.argmax([val[i%len(val)] for i in np.arange(np.argmax(zr)-int(len(val)/4),np.argmax(zr)+int(len(val)/4),1)]))%len(val)
    if plotrot :
        # plot radial that maximizes amplitude and correlation
        stm=rotate(st.copy(),angrange[bestind],correct)
        stn=stm.select(channel="*N")
        ste=stm.select(channel="*E")
        (stz+ste+stn).plot()
        if do_hilbert : stn[0].data=np.imag(hilbert(stn[0].data))
        stz=st.copy().select(channel="*Z")
        plt.imshow(env,interpolation='nearest',aspect='auto')
        scale = 10
        plt.plot(stn[0].data[start-win1:start+win2]/scale+180,c='red')
        plt.plot(stz[0].data[start-win1:start+win2]/scale+180,c='black')
        plt.plot(ste[0].data[start-win1:start+win2]/scale+180,c='orange')
        labels=['Radial','Vertical','Tangential']
        plt.legend(labels)
        plt.scatter(ind[np.argmax(val)],bestind,s=20,c='b')
        plt.show()
    # return: angle that maximizes ZR correlation, max correlation, angle that maximizes radial component and ZR correlation, correlation for that angle
    return angrange[np.argmax(zr)],np.max(zr),angrange[bestind],zr[bestind],(start-win1+ind[np.argmax(val)])/st[0].stats.sampling_rate


