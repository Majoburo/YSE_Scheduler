import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.time import Time,TimeDelta, TimezoneInfo
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun
from scipy.interpolate import interp1d
from matplotlib import cm
import argparse
import configparser


def parse_args():
    parser = argparse.ArgumentParser(description='Schedule targets for the night (FIRST DRAFT).')
    parser.add_argument('targetlist',
                        help = 'YSE-PZ format schedule with priority column as in example file.')
    parser.add_argument('instrument',
            help = 'Instrument. Options: KAST, NIRES')
    parser.add_argument('--date', dest='date',
                        type = Time,
                        help = 'fmt: yyyy-mm-dd. Date of observations. Superseeds taking the time from the filename.')
    parser.add_argument('--x', dest='x',
                        default = 1,
                        type = float,
                        help = 'Sets x in minimum(min, min^(1+x)*exptime^-x) where [exptime] = minutes')
    parser.add_argument('--min', dest='min',
                        default = 30,
                        type = float,
                        help = 'Sets min in minimum(min, min^(1+x)*exptime^-x) where [exptime] = minutes')
    parser.add_argument('--twimag', dest='twimag',
                        default = 15,
                        type = float,
                        help = 'Sets min magnitude for twilight targets.')
    parser.add_argument('--takesetting', dest='takesetting',
                        action='store_false',
                        help = 'Always take a target if setting.')
    parser.add_argument('--fluxstd', dest='fluxstd',
                        default="flux_std.dat",
                        type = str,
                        help = 'Append flux standards in file.')
    parser.add_argument('--manual', dest='manual',
                        action='store_true',
                        help = 'Manual mode, take all targets in the list order.')
    parser.add_argument('--start', dest='start',
                        type = Time,
                        help = 'fmt: yyyy-mm-ddThh:mm:ss (UTC). When to start observations')
    parser.add_argument('--end', dest='end',
                        type = Time,
                        help = 'fmt: yyyy-mm-ddThh:mm:ss (UTC). When to finish observations.')
    args = parser.parse_args()

    return args


def exposure_time(mag,data):
    # Calculates Exposure Times
    exp = interp1d(data[:,0],data[:,1],kind="nearest",fill_value="extrapolate")
    #n_blue = interp1d(data[:,0],data[:,2],kind="nearest",fill_value="extrapolate")
    #n_red = interp1d(data[:,0],data[:,3],kind="nearest",fill_value="extrapolate")
    #exp_red = exp(mag)*60//n_red(mag)
    #exp_blue = exp(mag)*60//n_blue(mag) + 15*np.ceil(n_red(mag)/n_blue(mag))
    return int(exp(mag))#, n_red(mag), exp_red, n_blue(mag), exp_blue

def fill_sched(schedu,df,start,end):
    entry = pd.DataFrame.from_dict({
        "target": [df["name"]],
        "RA":  ["%s:%s:%s"%(df["ra_h"],df["ra_m"],df["ra_s"])],
        "DEC": ["%s:%s:%s"%(df["dec_d"],df["dec_m"],df["dec_s"])],
        "start":[start],
        "end":[end],
        "mag":[df["mag"]]#,
        #"priority":[df["priority"]]
    })

    schedu = pd.concat([schedu, entry], ignore_index=True)
    return schedu

def set_exptime(target_list,manual,config,frame,times,inst):
    #Read in target list
    #df = pd.read_csv(target_list,delim_whitespace=True,skiprows=1, names= ["name","ra_h","ra_m","ra_s",'dec_d','dec_m','dec_s',"mag","priority"],usecols=[0,1,2,3,4,5,6,10,11])
    df = pd.read_csv(target_list,delim_whitespace=True,skiprows=1, names= ["name","ra_h","ra_m","ra_s",'dec_d','dec_m','dec_s',"mag"],usecols=[0,1,2,3,4,5,6,10])
    if not manual:
        df = df.sort_values(by=['ra_h','ra_m','ra_s'])
    radec = np.array(["%.0d:%.0d:%.2f %.0d:%.0d:%.2f"%(i[0],i[1],i[2],i[3],i[4],i[5]) for i in df[['ra_h','ra_m','ra_s','dec_d','dec_m','dec_s']].to_numpy()])
    targets = SkyCoord(radec,unit=(u.hourangle, u.deg))
    #For each target get the minimum airmass
    dts=[]
    for i,target in enumerate(targets):
        expdata = np.loadtxt(config[inst]['exp_file'])
        dt = TimeDelta(exposure_time(df.iloc[i]['mag'],expdata)*u.min)
        dts.append(dt)
    df["exp_time"] = np.array(dts)
    return df

def schedule(current_time,stop_time,df,target_list,schedu,dt,args):
    #priorities = list(set(df["priority"]))
    while current_time < stop_time:
        for j,target in df.iterrows():
            start = current_time
            current_time = current_time + target["exp_time"]
            end = current_time
            schedu = fill_sched(schedu,target,start,end)
            target_list.append(j)
            skipped_all=False
            current_time = current_time +dt
    return current_time,target_list,schedu

def gethour(time):
    PDT = (time - 8*u.hour).ymdhms
    PDT = "%02d:%02d"%(PDT[3],PDT[4])
    UTC = time.ymdhms
    utchour = float(UTC[3])+float(UTC[4])/60
    UTC = "%02d:%02d"%(UTC[3],UTC[4])
    return PDT,UTC, utchour

def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read('config.ini')
    inst = args.instrument
    # Establish observatory, sunset and twighlights
    obs = EarthLocation(lat=float(config[inst]['lat'])*u.deg,
            lon=float(config[inst]['lon'])*u.deg,
            height=float(config[inst]['height'])*u.m)

    utcoffset = -8*u.hour  # Assuming Pacific Time (Lick)
    if args.date:
        midnight = args.date - utcoffset
    else:
        try:
            date_str = args.targetlist.split("_")[1].split(".")[0]
            midnight = Time(date_str) - utcoffset
        except:
            print("File name does not include the date nor did you input a date in args. Please do one of the two.")

    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    frame = AltAz(obstime=times, location=obs)
    sunaltazs = get_sun(times).transform_to(frame)
    alt0 = np.diff((sunaltazs.alt.value>0))
    alt12 = np.diff((sunaltazs.alt.value>-12))
    alt18 = np.diff((sunaltazs.alt.value>-18))
    sunset, sunrise = times[1:][alt0]
    etwi12, mtwi12 = times[1:][alt12]
    etwi18, mtwi18 = times[1:][alt18]

    #Making a minutes timeframe for the night
    framestart= etwi12
    frameend= sunrise
    if args.start:
        framestart = args.start
    if args.end:
        frameend = args.end

    dt = frameend-framestart
    dt.format = 'sec'
    mins = round(dt.value/60)
    times = framestart + dt * np.linspace(0., 1., mins)
    frame = AltAz(obstime=times, location=obs)

    fluxtar = set_exptime(args.fluxstd,True,config,frame,times,inst)
    df = set_exptime(args.targetlist,args.manual,config,frame,times,inst)
    #df = pd.concat([fluxtar.iloc[:2],df,fluxtar.iloc[2:]])
    df = df.reset_index(drop=True)
    current_time = times[0]
    dt = TimeDelta(7*u.min)
    x = args.x
    target_list = []

    sched = pd.DataFrame(columns=["target","RA","DEC","start","end","mag"])#,"priority"])

    twitar = df[df["mag"]<args.twimag]
    #if args.fluxstd:
    #    current_time, target_list,sched = schedule(current_time,etwi18,df.iloc[:2],target_list,sched,dt,args)
    #current_time, target_list,sched = schedule(current_time,etwi18,twitar,target_list,sched,dt,args)
    current_time, target_list,sched = schedule(current_time,mtwi18,df,target_list,sched,dt,args)
    #print(current_time,mtwi12,twitar,df.iloc[2:])
    #current_time, target_list,sched = schedule(current_time,times[-1],twitar,target_list,sched,dt,args)
    #if args.fluxstd:
    #    current_time, target_list,sched = schedule(current_time,sunrise,df.iloc[2:],target_list,sched,dt,args)
    print(sched)


    filename = args.targetlist.split(".")[0]

    # Time to write things to a csv file

    with open(filename+"_Sched.csv", 'w') as f:
        f.write("Object,PDT,UTC,PDT End,UTC End,Ra,Dec,Exposure Time(s),Mag\n")
        sunsetPDT,sunsetUTC,sunsethour = gethour(sunset)
        sunrisePDT,sunriseUTC,sunrisehour = gethour(sunrise)
        mtwi12PDT, mtwi12UTC,mtw12hour = gethour(mtwi12)
        etwi12PDT, etwi12UTC,etw12hour = gethour(etwi12)
        mtwi18PDT, mtwi18UTC,mtw18hour = gethour(mtwi18)
        etwi18PDT, etwi18UTC,etw18hour = gethour(etwi18)
        f.write("Sunset,%s,%s\n"%(sunsetPDT,sunsetUTC))
        f.write("12 deg,%s,%s\n"%(etwi12PDT, etwi12UTC))
        f.write("18 deg,%s,%s\n"%(etwi18PDT, etwi18UTC))
        f.write("18 deg (morn),%s,%s\n"%(mtwi18PDT, mtwi18UTC))
        f.write("12 deg (morn),%s,%s\n"%(mtwi12PDT, mtwi12UTC))
        f.write("Sunrise,%s,%s\n"%(sunrisePDT,sunriseUTC))
        f.write("\n")
        for _,target in sched.iterrows():
            PDT, UTC,_ = gethour(target['start'])
            PDT_END, UTC_END,_ = gethour(target['end'])
            exp = round((target['end']- target['start']).sec)
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(target['target'],PDT,UTC,PDT_END,UTC_END,target['RA'],target['DEC'],exp,target['mag']))

    # Here be the plotting

    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_axes([0.12,0.15,0.7,0.7])
    style =["solid",(0,(5,1)),(0,(1,1))]
    for i,target in sched.iterrows():
        dt = target["end"] - target["start"]
        times = target["start"] + dt * np.linspace(0., 1., 100)
        frame = AltAz(obstime=times, location=obs)
        radec = "%s %s"%(target["RA"],target["DEC"])
        coords = SkyCoord(radec,unit=(u.hourangle, u.deg))
        targetaltazs = coords.transform_to(frame)
        plottime = np.array([tt[3]+tt[4]/60.+tt[5]/60./60. for tt in times.to_value("ymdhms")])*u.hour
        cmap = cm.get_cmap('Dark2')
        ax.plot(plottime, targetaltazs.secz,label=target["target"])#,linestyle=style[target['priority']-1])
    plt.axvline(x=sunsethour,color='orange')
    plt.axvline(x=sunrisehour,color='orange')
    plt.axvline(x=etw12hour)
    plt.axvline(x=mtw12hour)
    plt.axvline(x=etw18hour,color='black')
    plt.axvline(x=mtw18hour,color='black')
    ax.set_xlabel("UTC")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(["%02d:00"%x for x in ax.get_xticks()])
    axtop = ax.twiny()
    axtop.set_xticks(ax.get_xticks())
    axtop.set_xbound(ax.get_xbound())
    axtop.set_xticklabels(["%02d:00"%((x - 8)%12) for x in ax.get_xticks()])
    axtop.set_xlabel("PDT")
    ax.legend(bbox_to_anchor =(1.25, 1.0))
    ax.set_ylim(3, 0.9)
    ax.set_ylabel('Airmass')
    plot_lines=[]
    l1, = ax.plot(np.NaN,np.NaN, linestyle=style[0],color='black')
    l2, = ax.plot(np.NaN,np.NaN, linestyle=style[1],color='black')
    l3, = ax.plot(np.NaN,np.NaN, linestyle=style[2],color='black')

    plot_lines.append([l1, l2, l3])

    #legend1 = plt.legend(plot_lines[0], ["priority 1", "priority 2", "priority 3"], bbox_to_anchor =(1.25, 0.0))
    #plt.gca().add_artist(legend1)
    plt.savefig(filename+'_Sched.png')
    plt.show()



if __name__ == '__main__':
    main()


