import pygame, numpy as np
from pygame.locals import *
import random, sys, pickle
from scamp import Session, Ensemble, current_clock


# soundfonts from https://sites.google.com/site/soundfonts4u/home
generalSF = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
pianoSF = "/usr/share/sounds/sf2/Steinway_Grand-SF4U-v1.sf2"
essentialSF = "/usr/share/sounds/sf2/Essential_Keys-sforzando-v9.5.sf2"
bassSF = "/usr/share/sounds/sf2/Nice-Bass-Plus-Drums-v5.2.sf2"

if len(sys.argv)>=1+1+1:
    sf = sys.argv[1]
    std = sys.argv[2:]
else:
    std = ["Grand Piano"]
    sf = "/usr/share/sounds/sf2/SGM-v2.01-CompactGrand-Guit-Bass-v2.7.sf2"

number_of_counters_y = 7 # the logic depends on this number being 7, so when changing, one has to change the code
number_of_counters_x = len(std)

number_of_counters = number_of_counters_x*number_of_counters_y



from scamp import Session, Ensemble
from scamp._soundfont_host import get_best_preset_match_for_name

def construct_ensemble(sf):
    global piano_clef,piano_bass, flute, strings, session
    ensemble = Ensemble(default_soundfont=sf)

    ensemble.print_default_soundfont_presets()

    #print(len(sys.argv))
    #print(sys.argv)    
    #if len(sys.argv) == 8+1:
    #    std = sys.argv[1:]
    #else:
    #    std = ["Harp","Harp","Piano","Piano","Violin","Violoncello","Panflute","Acoustic Bass"] 
    #return [ensemble.new_midi_part(p,midi_output_device=6) for p in std] #lmms
    return [(ensemble.new_part(p),get_best_preset_match_for_name(p,which_soundfont=ensemble.default_soundfont)[0]) for p in std] #lmms
    #strings = ensemble.new_part("strings", (0, 40))

def aT(u,a):
    if u in [1,5,7,11]:
        return lambda x : (u*x+a)%12

def mul(U,V):
    u,a = U
    x,y = V
    return ((u*x)%12,(u*y+a)%12)

def iterMul(x,k):
    if k == 1:
        return x
    else:
        return mul(iterMul(x,k-1),x)

def orderMul(x):
    y = x
    o = 1
    while y!=(1,0):
        o+=1
        y = mul(y,x)
    return o






def drawRect(surface,color,xStart,yStart):
    pygame.draw.rect(surface, color, Rect(xStart,yStart,80,80),0)


def drawText(surface, text, x,y):
    textsurface = myfont.render(text, False, (255,255,255))
    surface.blit(textsurface,(x,y))

def getRect(x,y):
    k = (x//80)
    l = (y//80)
    return k+l*number_of_counters_x



oneOctave = list(range(60,72))
bassOctave = list(range(60-1*12,60-0*12))
startPitch = 0 
startPitchBass = 1
affineGroup = [aT(u,a) for u in [1,5,7,11] for a in range(12)]  
affineGroupIndex = [(u,a) for u in [1,5,7,11] for a in range(12)]  

affineGroupByOrder = [ (orderMul(x),x) for x in affineGroupIndex]
print(affineGroupByOrder)
twoLoops = [x for o,x in affineGroupByOrder if o==2]
threeLoops = [x for o,x in affineGroupByOrder if o==3]
fourLoops = [x for o,x in affineGroupByOrder if o==4]
print(len(twoLoops))
print(len(fourLoops))
countBass = 0



tempo = 120
s = Session(tempo=tempo,default_soundfont=sf).run_as_server()

s.print_available_midi_output_devices()

s.print_available_midi_input_devices()

tracks = construct_ensemble(sf)

print(len(tracks))

for t in tracks:
    s.add_instrument(t[0])

piano = s.new_part("Acoustic Bass")
piano = s.new_part("Concert Bass Drum")
s.add_instrument(piano)


def divisors(n):
    return [k for k in range(1,n+1) if n%k==0]

def play_piano(pitch,volume):
    global piano
    print("instrument:",pitch,volume)
    piano.start_note(pitch,volume)

def callback_midi(midi,dt):
    global s,piano, vol
    code,pitch,volume = midi
    print(midi,dt)
    if code == 144: # note on
        s.fork(play_piano,(pitch,127.0/(1+np.exp(-(volume-1)))))
    elif code==128: # note off
        piano.end_note()

#s.register_midi_listener(port_number_or_device_name=1, callback_function=callback_midi)


try:
    cc = pickle.load(open("counters_"+str(number_of_counters_x)+"x"+str(number_of_counters_y)+".pkl","rb"))
except:
    cc = None

if not cc is None and len(cc.keys())==number_of_counters:
    counters = cc
else:
    counters =  dict(zip(range(number_of_counters),number_of_counters*[0]))

def updateCounterForRect(rectangleNumber,plusOne=True):
    global counters
    if plusOne:
        counters[rectangleNumber] += 1
    else:
        counters[rectangleNumber] -= 1


def digits(n,base,padto=None):
    q = n
    ret = []
    while q != 0:
        q, r = q//base,q%base # Divide by 10, see the remainder
        ret.insert(0, r) # The remainder is the first to the right digit
    if padto is None:
        return ret
    for i in range(padto-len(ret)):
        ret.insert(0,0)
    #ret.extend((padto-len(ret))*[0])    
    return ret

def repeatingNumbers(dd,D,offset=1,NStart=1,NEnd = 100):
    return [sum([n//d for d in dd])%D+offset for n in range(NStart,NEnd+1)]

def durationMingus2Music21(mingusDuration):
    if float(mingusDuration) == 0.0:
        return None
    else:
        return float(1/mingusDuration)
    
def durationMingus2MidiUtil(mingusDuration)    :
    if float(mingusDuration) == 0.0:
        return None
    else:
        return float(1/mingusDuration*4) # beats per minutes in quarternotes

def digitsReversed(n,base,padto):
    q = n
    ret = []
    while q != 0:
        q, r = q//base,q%base # Divide by 10, see the remainder
        ret.insert(0, r) # The remainder is the first to the right digit
    if padto is None:
        return ret
    for i in range(padto-len(ret)):
        ret.insert(0,0)
    #ret.extend((padto-len(ret))*[0])    
    return ret


def sumTree(n,leftToRight=True):
    if n==1:
        return []
    else:
        if leftToRight:
            return [sumTree(int(n//2),leftToRight),sumTree(n-int(n//2),leftToRight)]
        else:
            return [sumTree(n-int(n//2),leftToRight),sumTree(int(n//2),leftToRight)]


def digitsTree(n):
    if n==0:
        return []
    dd = digits(n-1,2)
    dd.reverse()
    #print(dd)
    ll = []
    oo = [dd[i] for i in range(len(dd)) if i%2==1]
    ee = [dd[i] for i in range(len(dd)) if i%2==0]
    #print(oo,ee)
    O = sum([2**(i)*oo[i] for i in range(len(oo))])
    E = sum([2**(i)*ee[i] for i in range(len(ee))])
    #print(O,E)
    return [digitsTree(O),digitsTree(E)]


def getDurationsFromTree(tree):
    # Identify leaves by their length
    if len(tree) == 0:
        return [1]
    # Take all branches, get the paths for the branch, and prepend current
    dd = []
    for t in tree:
        dd.extend([2*d for d in getDurationsFromTree(t)])
    return dd 

def getDottedDurationsFromTree(tree,dotted=True):
    if len(tree)==0:
        return [1]
    dd = []
    if (len(tree[0])==0 or len(tree[1])==0) and dotted:
        dd.extend([4/3*d for d in getDottedDurationsFromTree(tree[0],dotted)])
        dd.extend([4*d for d in getDottedDurationsFromTree(tree[1],dotted)])
    else:
        for t in tree:
            dd.extend([2*d for d in getDottedDurationsFromTree(t,dotted)])
    return dd


def generateBar(nTracks,barNumber,notelist,SYMFUNC,NFUNC,BASEFUNC,vols,dotted):
    global number_of_counters_x, barCounter,counters,pitchCounter,maxN
    bars = []
    for i in range(nTracks):
        bars.append([])
        
    for tt in range(nTracks):
        for bb in [barNumber]:
            K = bb[tt]
            #print(K,barNumber)
            mingusDurations = getDottedDurationsFromTree([sumTree,digitsTree][counters[4*number_of_counters_x+tt]%2](max(K,1)),False)
            durations = [d for d in mingusDurations]
            pitches = []
            volumes = []
            if counters[1*number_of_counters_x+tt]>0 and counters[3*number_of_counters_x+tt]>0:
                c = barCounter[0]            
                barCounter[0] += 1
            else:
                c = bb[tt]
            for d in durations:
                c+= 1
                #print(tt,barCounter[0],c)
                dc = digitsReversed(c,BASEFUNC,padto=NFUNC)[:NFUNC]
                pitchMod = (SYMFUNC([d+1 for d in dc]))%len(notelist)
                print(dc,pitchMod)
                pitchlist = []
                if K>0: #and K%2==1: #>=mapToRest:
                    pitch = notelist[pitchMod]
                else:
                    pitch = None
                pitches.append([pitch])
                if vols == []:
                    volumes.append(0.5)
                else:
                    volumes.append(vols[tt]) #+counters[5*number_of_counters_x+tt]/10.0)    # todo: change this
            bar = list(zip(pitches,durations,volumes))        
            bars[tt].append(bar)    
    return(bars)

import math
SYMFUNC = lambda a : int(a[0]*a[1]*(a[0]+a[1])/int(math.gcd(a[0],a[1])**3))
NFUNC = 2
BASEFUNC = 19


import math
funcTirana = lambda a: 2*int(math.pow(a[2],2)) + 2*int(math.pow(a[3],2)) + 2*int(math.pow(a[4],2)) + 3*a[0] + 3*a[1]
funcTirana2 = lambda a: -(2*a[2]**2 + 2*a[3]**2 + 2*a[4]**2) + 3*a[0] + 3*a[1]
funcTirana3 = lambda a: (2*a[2]**2 + 2*a[3]**2 + 2*a[4]**2) + -(3*a[0] + 3*a[1])
funcTirana4 = lambda a: -(2*a[2]**2 + 2*a[3]**2 + 2*a[4]**2) + -(3*a[0] + 3*a[1])
funcABC = lambda a: int(a[0]*a[1]*(a[0]+a[1])/math.gcd(a[0],a[1])**3)
funcs = [funcTirana,funcTirana2,funcTirana3,funcTirana4]
funcKlein = lambda a : a[0]**1+a[1]**2+a[2]**1+a[3]**2
func = lambda a: sum(a) #int((a[0]**5*a[1]**1+a[0]**4*a[1]**2+a[0]**2*a[1]**4+a[0]**1*a[1]**5)/math.gcd(a[0],a[1])**6)
myfunc = func
NFUNC = 5
MFUNC = 6

listFuncs = [ 
               (funcTirana4,5,6),
               (funcABC, 2, 19),
               (funcKlein, 4, 6),
               (func,2,19),
               (func,3,10),
               (func,4,8),
               (func, 5,6),
               (funcTirana,5,6),
               (funcTirana2,5,6),
               (funcTirana3,5,6)
            ]

lfunc = [(funcABC,2),(funcKlein,4),(funcTirana,5)]

def play_bar_for_instrument(instNr,bar):
    global tracks, counters, number_of_counters_x,MyMIDI,timesInstr
    if counters[3*number_of_counters_x+instNr]<=0 or counters[instNr]<=0:
        return
    #print(tracks[instNr],bar)
    for i in range(len(bar)):
        nc,duration,volume = bar[i]
        dur = 4.0/(duration)
        if not nc[0] is None:
            pitch = max(min(nc[0]+counters[2*number_of_counters_x+instNr]*12,127),0) # octave at third row, counter = 0 -> 4-th octave
            tracks[instNr][0].play_note(pitch,volume, dur)
            MyMIDI.addNote(instNr, instNr, pitch, times[instNr] , dur, min(max(1,int(math.ceil(volume*127))),127))#*(ni-k+10.0)/(ni+10.0))
            times[instNr]+=dur
        
        

        




def setCounterToValue(rect,value):
    global counters
    counters[rect]= value

barCounter = dict(zip(range(len(tracks)),len(tracks)*[0]))
pitchCounter = dict(zip(range(len(tracks)),len(tracks)*[0]))


def tsp(data):
    import numpy as np
    from python_tsp.distances import great_circle_distance_matrix

    sources = np.array(data)
    distance_matrix = great_circle_distance_matrix(sources)
    from python_tsp.distances import tsplib_distance_matrix
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing

    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    return permutation, distance

def preprocess_inputs_kepler(df):
    import pandas as pd
    df = df.copy()
    
    # Drop unused columns
    df = df.drop(['rowid', 'kepid', 'kepler_name', 'koi_pdisposition', 'koi_score'], axis=1)
    
    # Limit target values to CANDIDATE and CONFIRMED
    false_positive_rows = df.query("koi_disposition == 'FALSE POSITIVE'").index
    df = df.drop(false_positive_rows, axis=0).reset_index(drop=True)
    
    # Drop columns with all missing values
    df = df.drop(['koi_teq_err1', 'koi_teq_err2'], axis=1)
    
    # Fill remaining missing values
    df['koi_tce_delivname'] = df['koi_tce_delivname'].fillna(df['koi_tce_delivname'].mode()[0])
    for column in df.columns[df.isna().sum() > 0]:
        if column == "kepoi_name":
            continue
        df[column] = df[column].fillna(df[column].mean())
    
    # One-hot encode koi_tce_delivname column
    delivname_dummies = pd.get_dummies(df['koi_tce_delivname'], prefix='delivname')
    df = pd.concat([df, delivname_dummies], axis=1)
    df = df.drop('koi_tce_delivname', axis=1)
    
    df = df.sample(64,random_state=42)
    
    # Split df into X and y
    y = df['kepoi_name'].values
    X = df.drop(['koi_disposition',"kepoi_name"], axis=1)
    return X,y

def slice_image(im,M=1,N=1):
    tiles = [im[x:x+M,y:y+N].flatten() for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    return tiles

def read_image(fn="./images/017.png_wave.jpg"):
    import cv2
    return cv2.imread(fn,1) # grayscale

maxN = 17

imageFN = "./image.jpg"
div = divisors(read_image(imageFN).shape[1]) # divisors of the length in y-direction of the image
nr = sorted([ (abs(d-50),d) for d in div])[0][1] # the closest divisor to the number 50

def write_images(img_folder="./img/"):
    global imageFN, nr
    import cv2, os

    fn = imageFN
    im = read_image(fn)
    print(im.shape)
    N = im.shape[0]
    M = im.shape[1]//nr
    counter = 0
    for x in range(0,im.shape[1],M):
        for y in range(0,im.shape[0],N):
            print((x,y),(x+M,y+N))
            tmp = im.copy()
            im2 = cv2.rectangle(tmp,(x,y), (x+M,y+N),(0,0,0),1)
            cv2.imwrite(img_folder+"img_"+"{0:05}".format(counter)+".png",im2)
            counter += 1

write_images()

def constructData():
    global tracks

    import numpy as np
    from sklearn.decomposition import PCA
    
    #import umap
    #reducer = umap.UMAP(n_neighbors=15,min_dist=0.1,n_components=2*len(tracks),metric="euclidean",random_state=42)
    pca = PCA(n_components=2*len(tracks))
    #from sklearn import datasets
    import pandas as pd

    import numpy as np
    from sklearn.impute import SimpleImputer
    #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

    #X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    #print(imp_mean.transform(X))    


    #mydata = pd.read_csv("/home/orges/kaggle/world_happines/2019.csv",sep=",",header=0) #datasets.load_boston()
    #mydata = pd.read_csv("/home/orges/kaggle/kepler/cumulative.csv",sep=",",header=0)
    #mydata,y = preprocess_inputs_kepler(mydata)
    #mydata = datasets.load_boston()
    #fn = "./milky_way/image.jpg"
    #nr = 86
    #fn = "./images/011.png_lazy.jpg"
    #nr = 512
    fn = imageFN
    #fn = "./300/image.jpg"
    #nr = 64
    im = read_image(fn)
    N = im.shape[1]//nr
    mydata = slice_image(im,M=im.shape[0],N=N)
    print(len(mydata))
    y = range(len(mydata))
    #X = mydata.iloc[:,2:]
    X = mydata #mydata.data
    X = pca.fit_transform(X)
    #y = mydata.iloc[:,1].values
    #y = mydata.target
    #print(y)
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    import numpy as np
    scaler = MinMaxScaler(feature_range=(1,maxN))
    data = [ tuple([int(x) for x in l]) for l in np.round(scaler.fit_transform(X),0).tolist()]
    #mapping = {0:"Setosa", 1: "Versicolour", 2: "Virginica"}
    print(data)
    return data,y

print("reading and transforming data...")
data,y = constructData()
index = 0
print("done.")

tsp = False

if tsp:
    print("tsp...")
    permutation,distance = tsp(data)
    data = [data[permutation[k]] for k in range(len(data))]
    y = [y[permutation[k]] for k in range(len(y))]
    print("done, tsp")

from midiutil import MIDIFile
import math
MyMIDI = MIDIFile(len(tracks),adjust_origin=False)
track = 0
time = 0

MyMIDI.addTempo(track,time, tempo)

track = 0
for k in range(len(tracks)):
    print( tracks[k][0].name)
    preset = tracks[k][1].preset
    track = k
    channel = k
    time = 0
    MyMIDI.addProgramChange(track,channel,time,preset) #alles erstmal auf piano=0 setzen

times = len(tracks)*[0]
toMusicXml = False

def scamp_loop():
   global countBass,tracks,oneOctave,SYMFUNC,NFUNC,BASEFUNC, counters, listFuncs,number_of_counters_x,s,started_transcribing, data, index, session_finished,toMusicXml
   #curr_clock = current_clock()
   #curr_clock.synchronization_policy = "no synchronization" #  http://scamp.marcevanstein.com/clockblocks/clockblocks.clock.Clock.html#clockblocks.clock.Clock
   while True:
      nTracks = len(tracks)
      barNumbers = data[index%len(data)][0:(len(tracks))] #[counters[k]  for k in range(nTracks)]
      vols = [v/(maxN*1.0) for v in data[index%len(data)][(len(tracks)):(2*len(tracks))]] #[counters[k]  for k in range(nTracks)]
      dotted = [] #data[index%len(data)][(2*len(tracks)):(3*len(tracks))]
      print(barNumbers,vols,dotted)
      #print(barNumbers)
      for k in range(len(tracks)): # update counters for visualisation for user
          counters[k] = barNumbers[k] 
      for tt in range(len(tracks)):
          #SYMFUNC,NFUNC,BASEFUNC = listFuncs[ max(counters[1*number_of_counters_x+tt],0)%len(listFuncs)]
          SYMFUNC,NFUNC = lfunc[counters[5*number_of_counters_x+tt]%len(lfunc)]
          BASEFUNC = max(2,counters[6*number_of_counters_x+tt])
          bars = generateBar(nTracks, barNumbers, oneOctave, SYMFUNC,NFUNC,BASEFUNC,vols,dotted)
          for t in range(len(bars[tt])):
              if counters[3*number_of_counters_x+tt]>0 and counters[tt]>0:
                  if not started_transcribing:
                      if toMusicXml:
                          s.start_transcribing()
                      started_transcribing = True
                  current_clock().fork(play_bar_for_instrument,(tt,bars[tt][t]))
                  #curr_clock.fork(play_bar_for_instrument,(tt,bars[tt][t]))
                  #curr_clock.wait_for_children_to_finish()
      if all([counters[3*number_of_counters_x+tt]>0 and counters[tt]>0 for tt in range(len(tracks))]): #all instruments played bar
          print("You were listening to  "+str(y[index%len(data)])+", a representation of the data by ", data[index%len(data)])
          if index == len(data):
               session_finished = True
               return
          index += 1 # increase the data counter
          
      if len(current_clock().children()) > 0:
          current_clock().wait_for_children_to_finish()
      #else:
      #    # prevents hanging if nothing has been forked yet
      #    curr_clock.wait(1.0)

description = ["BN","Cn","Octave","On/Off","Tree","FN","BF"]

session_finished = False

def main():
   global countBass,tracks,oneOctave,SYMFUNC,NFUNC,BASEFUNC, counters, listFuncs,number_of_counters_x, s,session_finished
   s_forked = False
   #s.fork(scamp_loop)

   while True:
      for event in pygame.event.get():
            xPos,yPos = (pygame.mouse.get_pos())
            leftPressed,middlePressed,rightPressed = pygame.mouse.get_pressed()
            #print(xPos,yPos,leftPressed,rightPressed)
            rect = getRect(xPos,yPos)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  #scroll up, increase counter by one
                    updateCounterForRect(rect,True)
                if event.button == 5: #scroll down, decrease counter by one
                    updateCounterForRect(rect,False)
            if leftPressed:
                val = counters[rect]//2 # left-button clicked: halve the value
                setCounterToValue(rect,val)
            if middlePressed:
                setCounterToValue(rect,0) # middle button clicked: set to zero
            if rightPressed:
                val =  counters[rect]*2 # right button clicked: double the value
                setCounterToValue(rect,val)
            #print(counters)
            if event.type == QUIT or session_finished:
               return
      for k in range(number_of_counters_x+1):
          for l in range(number_of_counters_y+1):
              if l<number_of_counters_y and k < number_of_counters_x:
                  color = ((k*5)%255,(l*5)%255,(k+l)%255)
                  drawRect(screen, color,k*80,l*80)
                  drawText(screen, str(counters[k+l*number_of_counters_x]), k*80+40,l*80+40)
              elif l == number_of_counters_y and k < number_of_counters_x:
                  color = (0,0,0)
                  drawRect(screen, color,k*80,l*80)
                  drawText(screen, tracks[k][0].name[0:3]+str("."), k*80+40,l*80+40) 
              elif l < number_of_counters_y and k == number_of_counters_x:
                  color = (0,0,0)
                  drawRect(screen, color,k*80,l*80)
                  drawText(screen, description[l][0:3]+str("."), k*80+40,l*80+40) 
              else:
                  drawRect(screen, color,k*80,l*80)

      pygame.display.flip()
      clock.tick()
      if any([counters[3*number_of_counters_x+tt]>0 and counters[tt]>0 for tt in range(len(tracks))]) and not s_forked: # first instrument starts playing
          #s.start_transcribing()
          s_forked = True
          s.fork(scamp_loop)

# Execute game:
started_transcribing = False

pygame.init()
#screen = pygame.display.set_mode((640, 480))
screen = pygame.display.set_mode((80*(number_of_counters_x+1), 80*(number_of_counters_y+1)))

clock = pygame.time.Clock()
pygame.font.init() # you have to call this at the start, 
                  # if you want to use this module.
myfont = pygame.font.SysFont('Comic Sans MS', 30)
#This creates a new object on which you can call the render method.

main()

# write counters to file:

with open("counters_"+str(number_of_counters_x)+"x"+str(number_of_counters_y)+".pkl","wb") as f:
    pickle.dump(counters,f)




if toMusicXml:
    performance = s.stop_transcribing()
    score = performance.to_score(time_signature="4/4")
    music_xml = score.to_music_xml()
    music_xml.export_to_file("my_music.xml")

with open("music.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

import os

command = "./Midi2mp3.sh "+sf+" music.mid" 
print(command)
os.system(command)

os.system("rm music.mp4")

command = "ffmpeg -r " +str(tempo/(4*60.0))+"  -pattern_type glob -i 'img/img_*.png' -i music.mp3  -acodec copy music.mp4"
print(command)
os.system(command)

os.system("rm img/img_*.png")
