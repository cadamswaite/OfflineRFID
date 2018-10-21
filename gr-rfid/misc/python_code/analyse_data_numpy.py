import scipy
import numpy as np
from os import getcwd



relative_path_to_file = '../data/Corilateme/source'
decim = 1 #decimation of matched filter
samp_rate = (2 * 10**6)/decim #Samples per second
half_symbol_length = int(round(12.5*10**-6*samp_rate))
print("Sample rate is ",samp_rate)
print("Half symbol length is ",half_symbol_length)
#Reduce computation by specifying a range to look at
first_sample = 46000
last_sample  = 100000

def decode_RN16(numpyarray):
    minimum = np.amin(numpyarray)
    maximum = np.amax(numpyarray)
    percentage_between = 20 #Should be 50, though lower value may give better results
    avg = np.average([minimum,maximum],weights=[100-percentage_between,percentage_between])
    zc = np.where(np.diff(np.sign(numpyarray-avg)))[0]
    xdiff = np.diff(zc)
    flt_xdiff = xdiff[xdiff>half_symbol_length/3.0]
    
    #apply sids algorithm (neglect first 8 datapoints)
    flt_xdiff = flt_xdiff[7:np.where(flt_xdiff>2*half_symbol_length)[0][-1]]
    print(xdiff)
    print(flt_xdiff)
    
    RN16_bits=[]
    last_midpoint_state = 1 
    for x in np.nditer(flt_xdiff):
        print("AT start of loop, last_m was ",last_midpoint_state)
        if last_midpoint_state == 0:
            last_midpoint_state = 1
            RN16_bits.append(0)
            print("Appended a 0")
            continue
        print(x)
        if x < 1.3*half_symbol_length:
            print("probs a 0")
            last_midpoint_state = 0
        else:
            RN16_bits.append(1)
    print(RN16_bits,len(RN16_bits))
        
    
    
    #TODO implement more efficiently using numpy
    #position of 1s
    print(np.where(flt_xdiff>1.3*half_symbol_length)[0])
    print([22]+np.diff(np.where(flt_xdiff>1.3*half_symbol_length)[0])-1)
    
    #TODO if the length is longer than a fullbit, likely 
    #print(np.where(flt_xdiff>2.5*half_symbol_length)[0])
    print(int(7/2.0))
    print("Max and min are ",minimum,maximum)
    #plt.axhline(y=avg, color='r', linestyle='-')

f = scipy.fromfile(open(getcwd()+'/'+relative_path_to_file), dtype=scipy.float32)
print("Number of datapoints is:",f.size)
f=f[first_sample:last_sample]
abs_f=abs(f[0::2]+1j*f[1::2])



#import matplotlib.pyplot as plt
#plt.plot(abs_f)
decode_RN16(abs_f[9800:11200])
#plt.show()

