import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.time import Time,TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun
from scipy.interpolate import interp1d
from matplotlib import cm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Schedule targets for the night for Shane (FIRST DRAFT).')
    parser.add_argument('targetlist',
                        help = 'YSE-PZ format schedule with priority column as in example file.')
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
    args = parser.parse_args()

    return args


def exposure_time(mag):
    #Calculates Exposure Times
    data = np.loadtxt("exposure_kast.dat")
    exp = interp1d(data[:,0],data[:,1],kind="nearest",fill_value="extrapolate")
    n_blue = interp1d(data[:,0],data[:,2],kind="nearest",fill_value="extrapolate")
    n_red = interp1d(data[:,0],data[:,3],kind="nearest",fill_value="extrapolate")
    exp_red = exp(mag)*60//n_red(mag)
    exp_blue = exp(mag)*60//n_blue(mag) + 15*np.ceil(n_red(mag)/n_blue(mag))
    return int(exp(mag)), n_red(mag), exp_red, n_blue(mag), exp_blue


def main():
    args = parse_args()
    # Establish observatory, sunset and twighlights
    lick_obs = EarthLocation(lat=37.341389*u.deg, lon=-121.642778*u.deg, height=1290*u.m)
    utcoffset = -8*u.hour  # Assuming Pacific Time (Lick)
    if args.date:
        midnight = args.date - utcoffset
    else:
        date_str = args.targetlist.split("_")[1].split(".")[0]
        midnight = Time(date_str) - utcoffset

    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    frame = AltAz(obstime=times, location=lick_obs)
    sunaltazs = get_sun(times).transform_to(frame)
    alt0 = np.diff((sunaltazs.alt.value>0))
    alt12 = np.diff((sunaltazs.alt.value>-12))
    alt18 = np.diff((sunaltazs.alt.value>-18))
    sunset, sunrise = times[1:][alt0]
    etwi12, mtwi12 = times[1:][alt12]
    etwi18, mtwi18 = times[1:][alt18]

    #Making a minutes timeframe for the night
    dt = mtwi12 - etwi12
    dt.format = 'sec'
    mins = round(dt.value/60)
    times = etwi12 + dt * np.linspace(0., 1., mins)
    frame = AltAz(obstime=times, location=lick_obs)

    #Read in target list
    df = pd.read_csv(args.targetlist,delim_whitespace=True,skiprows=1, names= ["name","ra_h","ra_m","ra_s",'dec_d','dec_m','dec_s',"mag","priority"],usecols=[0,1,2,3,4,5,6,10,11])
    df = df.sort_values(by=['ra_h','ra_m','ra_s'])
    radec = np.array(["%.0d:%.0d:%.2f %.0d:%.0d:%.2f"%(i[0],i[1],i[2],i[3],i[4],i[5]) for i in df[['ra_h','ra_m','ra_s','dec_d','dec_m','dec_s']].to_numpy()])
    targets = SkyCoord(radec,unit=(u.hourangle, u.deg))
    #For each target get the minimum airmass
    airmasses=[]
    dts=[]
    alts=[]
    set_time=[]
    for i,target in enumerate(targets):
        airmass = target.transform_to(frame).secz
        mask = airmass<1
        airmass[mask] = 999
        airmasses.append(airmass)
        if min(airmass)>2.5:
            print("%s has a minimum airmass higher than 2.5!"%df.iloc[i]['name'])
            #eliminate target here TODO
        settime = times[:-1][np.diff((airmass>=3)*1)==1]
        if len(settime)>0:
            set_time.append(settime[0])
        else:
            set_time.append(times[-1])
        alts.append(times[np.argmin(airmass)])
        dt = TimeDelta(exposure_time(df.iloc[i]['mag'])[0]*u.min)
        dts.append(dt)
    df["exp_time"] = np.array(dts)
    df["top_time"] = np.array(alts)
    df["set_time"] = np.array(set_time)
    airmasses = np.array(airmasses)
    # sorting by setting time and the ones that don't set, sorting by RA
    #import pdb
    #pdb.set_trace()
    #argtargets,argtime = np.where(np.diff((airmasses>=3)*1)==1)
    #a = argtargets[np.argsort(argtime)]
    #df.index = np.append(a,list(set(np.arange(len(df))).difference(a)))
    #df.sort_index(inplace=True)
    #df.index.name = "settingorder"
    df.sort_values(by=['priority','top_time'],ascending=[True,True],inplace=True)
    def is_observable(time,dt,setting):
        if time+dt < setting:
            return True
        else:
            return False
    def is_setting(time,toptime):
        if time >= toptime:
            return True
        else:
            return False

    current_time = times[0]
    dt = TimeDelta(5*u.min)
    x = args.x
    target_list = []

    sched = pd.DataFrame(columns=["target","RA","DEC","start","end","mag","priority"])

    def fill_sched(schedu,df,start,end):
        entry = pd.DataFrame.from_dict({
            "target": [df["name"]],
            "RA":  ["%s:%s:%s"%(df["ra_h"],df["ra_m"],df["ra_s"])],
            "DEC": ["%s:%s:%s"%(df["dec_d"],df["dec_m"],df["dec_s"])],
            "start":[start],
            "end":[end],
            "mag":[df["mag"]],
            "priority":[df["priority"]]
        })

        schedu = pd.concat([schedu, entry], ignore_index=True)
        return schedu

    def schedule(current_time,stop_time,df,target_list,schedu):
        priorities = list(set(df["priority"]))
        while current_time < stop_time:
            skipped_all=True
            for i,target in df.iterrows():
                if i in target_list:
                    continue
                boundtime = min(args.min, args.min**(1+x)*(target["exp_time"].sec/60)**(-x)) + target["exp_time"].sec/60/2
                boundtime = boundtime*u.min
                if is_observable(current_time,target["exp_time"],target["set_time"]):
                    if args.takesetting:
                        timediff = abs(target["top_time"] - current_time)
                    else:
                        timediff = target["top_time"] - current_time
                    if boundtime > timediff:
                        start = current_time
                        current_time = current_time + target["exp_time"]
                        end = current_time
                        schedu = fill_sched(schedu,target,start,end)
                        target_list.append(i)
                        skipped_all=False
                        current_time = current_time +dt
                        #print(target['name'])
                        #print(boundtime)#-target["exp_time"].sec/60/2*u.min)
                        break
            if skipped_all:
                current_time = current_time + dt
        return current_time,target_list,schedu

    twitar = df[df["mag"]<args.twimag]
    current_time, target_list,sched = schedule(current_time,etwi18,twitar,target_list,sched)
    current_time, target_list,sched = schedule(current_time,mtwi18,df,target_list,sched)
    current_time, target_list,sched = schedule(current_time,times[-1],twitar,target_list,sched)
    print(sched)

    def gethour(time):
        PDT = (time - 8*u.hour).ymdhms
        PDT = "%02d:%02d"%(PDT[3],PDT[4])
        UTC = time.ymdhms
        utchour = float(UTC[3])+float(UTC[4])/60
        UTC = "%02d:%02d"%(UTC[3],UTC[4])
        return PDT,UTC, utchour

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
        frame = AltAz(obstime=times, location=lick_obs)
        radec = "%s %s"%(target["RA"],target["DEC"])
        coords = SkyCoord(radec,unit=(u.hourangle, u.deg))
        targetaltazs = coords.transform_to(frame)
        plottime = np.array([tt[3]+tt[4]/60.+tt[5]/60./60. for tt in times.to_value("ymdhms")])*u.hour
        cmap = cm.get_cmap('Dark2')
        ax.plot(plottime, targetaltazs.secz,label=target["target"],linestyle=style[target['priority']-1])
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

    legend1 = plt.legend(plot_lines[0], ["priority 1", "priority 2", "priority 3"], bbox_to_anchor =(1.25, 0.0))
    plt.gca().add_artist(legend1)
    plt.savefig(filename+'_Sched.png')
    plt.show()



if __name__ == '__main__':
    main()


