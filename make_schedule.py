import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.time import Time,TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun
from scipy.interpolate import interp1d
import argparse
from datetime import date



parser = argparse.ArgumentParser(description='Schedule targets for the night (FIRST DRAFT).')
parser.add_argument('targetlist',
                    help='YSE-PZ format schedule')
#parser.add_argument('--date', dest='date',
##                    type=date.fromisoformat,
#                    help='Date of observations. Defaults to today.')

args = parser.parse_args()





#Calculate Exposure Times
data = np.loadtxt("exposure_kast.dat")
def exposure_time(mag):
    exp = interp1d(data[:,0],data[:,1],kind="nearest",fill_value="extrapolate")
    n_blue = interp1d(data[:,0],data[:,2],kind="nearest",fill_value="extrapolate")
    n_red = interp1d(data[:,0],data[:,3],kind="nearest",fill_value="extrapolate")
    exp_red = exp(mag)*60//n_red(mag)
    exp_blue = exp(mag)*60//n_blue(mag) + 15*np.ceil(n_red(mag)/n_blue(mag))
    return int(exp(mag)), n_red(mag), exp_red, n_blue(mag), exp_blue


#Establish observatory, sunset and twighlights
lick_obs = EarthLocation(lat=37.341389*u.deg, lon=-121.642778*u.deg, height=1290*u.m)
utcoffset = -8*u.hour  # Pacific Time
midnight = Time('2022-3-06 00:00:00') - utcoffset
#midnight = args.date - utcoffset
delta_midnight = np.linspace(-12, 12, 1000)*u.hour
times = midnight + delta_midnight
frame = AltAz(obstime=times, location=lick_obs)
sunaltazs = get_sun(times).transform_to(frame)
alt0 = np.diff((sunaltazs.alt.value>0))
alt12 = np.diff((sunaltazs.alt.value>-12))
alt18 = np.diff((sunaltazs.alt.value>-18))
sunset, sunrise = times[1:][alt0]
mtwi12, etwi12 = times[1:][alt12]
mtwi18, etwi18 = times[1:][alt18]

#Making a minutes timeframe for the night
#dt = etwi18 - mtwi18
dt = sunrise - sunset
dt.format = 'sec'
mins = round(dt.value/60)
#print(mins/60)
times = sunset + dt * np.linspace(0., 1., mins)
#print(etwi18,times[-1])
#times = sunset + dt * np.linspace(0., 1., mins)

frame = AltAz(obstime=times, location=lick_obs)

#Read in target list
df = pd.read_csv(args.targetlist,delim_whitespace=True,skiprows=1, names= ["name","ra_h","ra_m","ra_s",'dec_d','dec_m','dec_s',"mag"],usecols=[0,1,2,3,4,5,6,10])
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
# sorting by setting time and the ones that don't set, keeping by RA
argtargets,argtime = np.where(np.diff((airmasses>=3)*1)==1)
a = argtargets[np.argsort(argtime)]
df.index=np.append(a,list(set(np.arange(18)).difference(a)))
df.sort_index(inplace=True)

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
x=1
target_list = []

sched = pd.DataFrame(columns=["target","RA","DEC","start","end"])
def fill_sched(sched,df,start,end):
    entry = pd.DataFrame.from_dict({
        "target": [df["name"]],
        "RA":  ["%s:%s:%s"%(df["ra_h"],df["ra_m"],df["ra_s"])],
        "DEC": ["%s:%s:%s"%(df["dec_d"],df["dec_m"],df["dec_s"])],
        "start":[start],
        "end":[end],
        "mag":[df["mag"]]
    })

    sched = pd.concat([sched, entry], ignore_index=True)
    return sched

while current_time < times[-1]:
    skipped_all=True
    for i,target in df.iterrows():

        if i in target_list:
            continue
        if is_setting(current_time,target["top_time"]):
            if is_observable(current_time,target["exp_time"],target["set_time"]):
                start = current_time
                current_time = current_time + target["exp_time"]
                end = current_time
                sched = fill_sched(sched,target,start,end)
                target_list.append(i)
                skipped_all=False
                current_time = current_time +dt
                break
            else:
                continue
        elif current_time + x*target["exp_time"]> target["top_time"]:
            start = current_time
            current_time = current_time + target["exp_time"]
            end = current_time
            sched = fill_sched(sched,target,start,end)
            target_list.append(i)
            skipped_all=False
            current_time = current_time +dt
            break
    if skipped_all:
        current_time = current_time + dt

def gethour(time):
    PDT = (time - 8*u.hour).ymdhms
    PDT = "%s:%s"%(PDT[3],PDT[4])
    UTC = time.ymdhms
    UTC = "%s:%s"%(UTC[3],UTC[4])
    return PDT,UTC

with open("Schedule.csv", 'w') as f:
    f.write("Object,PDT,UTC,PDT End,UTC End,Ra,Dec,Exposure Time(s),Mag\n")
    sunsetPDT,sunsetUTC = gethour(sunset)
    sunrisePDT,sunriseUTC = gethour(sunrise)
    mtwi12PDT, mtwi12UTC = gethour(mtwi12)
    etwi12PDT, etwi12UTC = gethour(etwi12)
    mtwi18PDT, mtwi18UTC = gethour(mtwi12)
    etwi18PDT, etwi18UTC = gethour(etwi12)
    f.write("Sunset,%s,%s\n"%(sunsetPDT,sunsetUTC))
    f.write("12 deg,%s,%s\n"%(etwi12PDT, etwi12UTC))
    f.write("18 deg,%s,%s\n"%(etwi18PDT, etwi18UTC))
    f.write("18 deg (morn),%s,%s\n"%(mtwi18PDT, mtwi18UTC))
    f.write("12 deg (morn),%s,%s\n"%(mtwi12PDT, mtwi12UTC))
    f.write("Sunrise,%s,%s\n"%(sunrisePDT,sunriseUTC))
    f.write("\n")
    for _,target in sched.iterrows():
        PDT = (target['start'] - 8*u.hour).ymdhms
        PDT = "%s:%2d"%(PDT[3],PDT[4])
        UTC = target['start'].ymdhms
        UTC = "%s:%2d"%(UTC[3],UTC[4])
        PDT_END = (target['end'] - 8*u.hour).ymdhms
        PDT_END = "%s:%2d"%(PDT_END[3],PDT_END[4])
        UTC_END = target['end'].ymdhms
        UTC_END = "%s:%2d"%(UTC_END[3],UTC_END[4])
        exp = round((target['end']- target['start']).sec)
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(target['target'],PDT,UTC,PDT_END,UTC_END,target['RA'],target['DEC'],exp,target['mag']))


for i,target in sched.iterrows():
    dt = target["end"] - target["start"]
    times = target["start"] + dt * np.linspace(0., 1., 100)
    frame = AltAz(obstime=times, location=lick_obs)
    radec = "%s %s"%(target["RA"],target["DEC"])
    coords = SkyCoord(radec,unit=(u.hourangle, u.deg))
    targetaltazs = coords.transform_to(frame)
    plt.plot(times.jd, targetaltazs.secz,label=target["target"])
plt.legend(loc='upper right')
plt.ylim(3, 1)
plt.xlabel('julian date (sorry)')
plt.ylabel('Airmass')
plt.show()