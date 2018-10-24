import scipy
from scipy.signal import argrelextrema
import numpy as np
from os import getcwd



relative_path_to_file = '../data/Corilateme/source'
decim = 1 #decimation of matched filter
samp_rate = (2 * 10**6)/decim #Samples per second
half_symbol_length = int(round(12.5*10**-6*samp_rate))
print("Sample rate is ",samp_rate)
print("Half symbol length is ",half_symbol_length)
#Reduce computation by specifying a range to look at
first_sample = 0
last_sample  = 1400000
verbose = False
plotit = False

def gate_signal(numpyarray):
    '''Removes the high amplitude signals sent Transmitter->receiver.
    The output signal can then be fed into find_RN16 to find where commands are sent.
    '''
    
    return numpyarray
    
    

def find_RN16(numpyarray):
    '''Find the preamble of RN16 using cross-correlation'''
    #TODO make this correct for different sample rates
    signal = [1,0,1,0,2]
    #TODO downsample for speed
    sampled_signal = np.concatenate((50*[1],25*[-1],25*[1],50*[-1],25*[1],75*[-1],25*[1]))
    #flipped = np.flipud(sampled_signal)
    correlated = np.correlate(numpyarray-np.mean(numpyarray),sampled_signal)
    a = np.where(correlated>0.2)
    start_locations = np.take(a,argrelextrema(correlated[a], np.greater)[0])
    #print("RN16 start locs are",start_locations)
    return start_locations
    
    
    
def find_initial_transmissions(numpyarray):
    
    #TODO allow this to change with frequency
    ds = 5
    downsampled = numpyarray[::ds]
    start_transmit = np.concatenate((1000/ds*[1],25/ds*[-1],25/ds*[1],25/ds*[-1],50/ds*[1]))
    correlated = np.correlate(downsampled-0.5,start_transmit)
    a = np.where(correlated>530/ds)
    start_locations = np.take(a,argrelextrema(correlated[a], np.greater)[0])*ds + 995 #Since started sampling 1000 before the real signal
    # Finishes roughly 1800 later

    return start_locations



def decode_RN16(numpyarray,remove,pie):
    '''Decodes the array into bits
    Arguments:
    numpyarray is an array containing the data to be processed
    remove is the number of bits at the start that should be ignored.
        common for preamble
    pie switches the mode from FM0 decoding to PIE.
        FM0 is used for tag to reader
        PIE is used for reader to tag
      
    Allows this function to do RN16 and ACK decoding.
    
    '''
    percentage_between = 20 #Should be 50, though lower value may give better results
    # Perform a weighted mean between the maximum and minimum values.
    # Due to clipping, 50% occurs quite high up, causing some false transitions.
    avg = np.average([np.amin(numpyarray),np.amax(numpyarray)],weights=[100-percentage_between,percentage_between])
    zc = np.where(np.diff(np.sign(numpyarray-avg)))[0]
    xdiff = np.diff(zc)
    flt_xdiff = xdiff[xdiff>half_symbol_length/3.0]
    if verbose:
        print(flt_xdiff)
    flt_xdiff = flt_xdiff[remove:]
    #Differential decoder
    RN16_bits=[]
    mid_read = False
    for x in np.nditer(flt_xdiff):
        if mid_read:
            mid_read = False
            continue
        if x < 1.3*half_symbol_length:
            RN16_bits.append(0)
            mid_read = True
        else:
           RN16_bits.append(1)
           if pie:
               mid_read = True
    
    #TODO implement more efficiently using numpy
    #position of 1s
    #print(np.where(flt_xdiff>1.3*half_symbol_length)[0])
    #print(np.diff(np.where(flt_xdiff>1.3*half_symbol_length)[0])-1)
    if verbose:
        if pie:    
            print("ACK  is ",RN16_bits,len(RN16_bits))
        else:    
            print("RN16 is ",RN16_bits,len(RN16_bits))

    
    if plotit:
        print("Max and min are ",np.amin(numpyarray),np.amax(numpyarray))
        plt.axhline(y=avg, color='r', linestyle='-')
    return RN16_bits


    
def crc16(data, poly=0x8408):
    #'''Function adapted from https://www.cgran.org/wiki/Gen2'''
    #data = np.zeros(128)
    #crc_16 = 0xFFFF
    #for i in xrange(len(bits)):
    #    for j in xrange(8):
    #        if bits[8*i+j]=='1':
    #            
    #    #print(i)
    #    pass
    pass
    
#PC + EPC + CRC16 + Dummy = 6 + 16 + 96 + 16 + 1 = 135
def decode_epc(nparray):
    bits = decode_RN16(nparray,0,0)
    EPC = bits[22:118]
    
    #print('EPC IS :',''.join(map(str,EPC)),len(EPC))
    #print(crc16(EPC))
    print(''.join(map(str,bits)),len(bits))



f = scipy.fromfile(open(getcwd()+'/'+relative_path_to_file), dtype=scipy.float32)
print("Number of datapoints is:",f.size)
f=f[first_sample:last_sample]
abs_f=abs(f[0::2]+1j*f[1::2])

import matplotlib.pyplot as plt

plt.plot(abs_f)
#plt.plot(
transmission_starts = find_initial_transmissions(abs_f)
y = np.ones(len(transmission_starts))*1.06
plt.scatter(transmission_starts,y,c='r',marker='x')

start_RN16=[]
print("startrn16 is ",start_RN16)
for x in range(len(transmission_starts)-1):
    relative_start_loc = find_RN16(abs_f[transmission_starts[x]+1800:transmission_starts[x+1]])
    try:
        start_RN16.append(transmission_starts[x]+1800 + relative_start_loc[0])
    except IndexError:
        pass
#print(start_RN16)

print("Number of transmissions",len(transmission_starts))
print("Sum of RN16s + EPCs",len(start_RN16))

y = np.ones(len(start_RN16))*1.06
plt.scatter(start_RN16,y,c='b',marker='x')
plt.show()

if plotit:
    import matplotlib.pyplot as plt
    plt.plot(abs_f)
    decode_RN16(abs_f[43000:44500],6,0)
    decode_RN16(abs_f[45000:46600],8,1)
    plt.show()
else:
    print("RN16 is ",decode_RN16(abs_f[43000:44500],7,0))
    print("ACK  is ",decode_RN16(abs_f[45000:46600],9,1))
    decode_epc(abs_f[61800:68800])
    decode_epc(abs_f[76800:83800])

