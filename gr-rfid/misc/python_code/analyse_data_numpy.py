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
first_sample = 100000
last_sample  = 200000
verbose = False
plotit = False


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
    percentage_between = 30 #Should be 50, though lower value may give better results
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
    
    if pie:    
        print("ACK  is ",RN16_bits,len(RN16_bits))
    else:    
        print("RN16 is ",RN16_bits,len(RN16_bits))

    
    if plotit:
        print("Max and min are ",np.amin(numpyarray),np.amax(numpyarray))
        plt.axhline(y=avg, color='r', linestyle='-')



f = scipy.fromfile(open(getcwd()+'/'+relative_path_to_file), dtype=scipy.float32)
print("Number of datapoints is:",f.size)
f=f[first_sample:last_sample]
abs_f=abs(f[0::2]+1j*f[1::2])


if plotit:
    import matplotlib.pyplot as plt
    plt.plot(abs_f)
    decode_RN16(abs_f[43000:44500],7,0)
    decode_RN16(abs_f[45000:46600],9,1)
    plt.show()
else:
    decode_RN16(abs_f[43000:44500],7,0)
    decode_RN16(abs_f[45000:46600],9,1)

