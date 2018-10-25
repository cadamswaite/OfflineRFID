import scipy
from scipy.signal import argrelextrema
import numpy as np
from os import getcwd



relative_path_to_file = '../data/Corilateme/source'
relative_path_to_file = '../data/file_source_test'
decim = 5 #decimation of matched filter
samp_rate = (10 * 10**6)/decim #Samples per second
half_symbol_length = int(round(12.5*10**-6*samp_rate))
print("Sample rate is ",samp_rate)
print("Half symbol length is ",half_symbol_length)
#Reduce computation by specifying a range to look at
first_sample = 70000
last_sample  = 14000000
verbose = False
plotit = False

def find_RN16(numpyarray):
    '''Find the preamble of RN16 using cross-correlation'''
    #TODO downsample for speed, if reliable enough.
    sampled_signal = np.concatenate((2*half_symbol_length*[1],half_symbol_length*[-1],half_symbol_length*[1],
                                                            2*half_symbol_length*[-1],half_symbol_length*[1],
                                                            3*half_symbol_length*[-1],half_symbol_length*[1]))
    #flipped = np.flipud(sampled_signal) #Usefull if convolving
    correlated = np.correlate(numpyarray-np.mean(numpyarray),sampled_signal)
    start_location = np.argmax(correlated)
    #plt.plot(correlated)
    #plt.show()
    #print("RN16 start loc is",start_location)
    return start_location
    
def find_start_transmission(numpyarray):
    '''Find the preamble of Transmission using cross-correlation'''
    prev_samples = 700
    start_transmit = np.concatenate((prev_samples*[1],half_symbol_length*[-1],half_symbol_length*[1],
                                                      half_symbol_length*[-1],2*half_symbol_length*[1]))    
    # Normalise, so midpoint is known.
    norm_numpyarray = numpyarray/np.amax(numpyarray)
    correlated = np.correlate(norm_numpyarray-0.5,start_transmit)
    start_location = np.argmax(correlated)+ prev_samples
    #plt.plot(correlated)
    #plt.show()
    #print("Transmission start loc is",start_location)
    return start_location  

def find_end_transmission(numpyarray):
    '''Find the last bit location of transmission using cross-correlation'''
    following_samples = 300
    start_transmit = np.concatenate((half_symbol_length*[-1],following_samples*[1]))
    norm_numpyarray = numpyarray/np.amax(numpyarray)
    correlated = np.correlate(norm_numpyarray-0.5,start_transmit)
    start_location = np.argmax(correlated)+half_symbol_length
    #plt.plot(correlated)
    #plt.show()
    #print("Transmission end loc is",start_location)
    return start_location                                     
    
#TODO remove when working state machine.
def find_initial_transmissions(numpyarray):
    
    #TODO allow this to change with frequency
    ds = 1
    downsampled = numpyarray[::ds]
    prev_samples = 700
    start_transmit = np.concatenate((prev_samples/ds*[1],half_symbol_length/ds*[-1],half_symbol_length/ds*[1],
                                                         half_symbol_length/ds*[-1],2*half_symbol_length/ds*[1]))
    
    #Assuming transmit>>received, normalising should make midpoint 0.5, thus 0.5 is not magic.
    norm_downsampled = downsampled/np.amax(downsampled)
    correlated = np.correlate(norm_downsampled-0.5,start_transmit)
    #plt.plot(correlated)
    #plt.show()
    
    #TODO remove magic number 
    # Need to find a way to find all of the high peaks for the given data.
    # Maybe generate the magicnumber dynamically eg correlated>max(correlated)-10
    # Since the transmissions will always be (roughly) the same amplitude after scaling, might not need to.
    a = np.where(correlated>325)

    #print("Correlated is",correlated)
    start_locations = np.take(a,argrelextrema(correlated[a], np.greater)[0])*ds + prev_samples #Since started sampling 500 before the real signal
    # Finishes roughly 1800 later
    print("Transmission start loc is ", start_locations)
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
    percentage_between = 50 #Should be 50, though lower value may give better results
    # Perform a weighted mean between the maximum and minimum values.
    # Due to clipping, 50% occurs quite high up, causing some false transitions.
    avg = np.average([np.amin(numpyarray),np.amax(numpyarray)],weights=[100-percentage_between,percentage_between])
    zc = np.where(np.diff(np.sign(numpyarray-avg)))[0]
    xdiff = np.diff(zc)
    # Filter out the short duration pulses, less than 1/3 of the smallest possible pulse is definitely an error.
    # Could filter more strictly in the future, or do something clever like combining small (erronious) pulses with their 
    # neighbours so the total adds to a plausable symbol_length.
    flt_xdiff = xdiff[xdiff>half_symbol_length/3.0]
    print("Len before and after filtering",len(xdiff),len(flt_xdiff))
    if verbose:
        print("flt_xdiff IS ",flt_xdiff)
    flt_xdiff = flt_xdiff[remove:]
    #Differential decoder
    #Using sidarth's method of counting zero crossings. Works well with high snr, but has major issues if
    #Bounce can be mistaken for a crossing.
    #Best to re-implement using Nikos' method of sampling at known times
    RN16_bits=[]
    mid_read = False
    try:
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
    except ValueError:
        return [0]
    
    #TODO implement more efficiently using numpy
    #position of 1s
    #print(np.where(flt_xdiff>1.3*half_symbol_length)[0])
    #print(np.diff(np.where(flt_xdiff>1.3*half_symbol_length)[0])-1)
    if verbose:
        if pie:    
            print("ACK  is ",RN16_bits,len(RN16_bits))
        else:    
            print("RN16 is ",RN16_bits,len(RN16_bits))
    # Can have 17bits read, last one is false.
    if len(RN16_bits)==17:
        RN16_bits = RN16_bits[:16]
    
    if plotit:
        print("Max and min are ",np.amin(numpyarray),np.amax(numpyarray))
        plt.axhline(y=avg, color='r', linestyle='-')
    return RN16_bits



#File operations
f = scipy.fromfile(open(getcwd()+'/'+relative_path_to_file), dtype=scipy.float32)
print("Number of datapoints is:",f.size)
f=f[first_sample:last_sample]
abs_f=abs(f[0::2]+1j*f[1::2])

## Matched filter to reduce hf noise
abs_f = scipy.signal.correlate(abs_f, np.ones(decim), mode='same') / decim

import matplotlib.pyplot as plt
def find_valid_transmission(relative_position):
    '''
    Recursive function to find a valid start point. May eat memory.
    '''

    first_tran_start = find_start_transmission(abs_f[0+relative_position:10000+relative_position])+relative_position
    #Searches for the end between 1500 and 2500 later than the start.
    min_trans,max_trans = 1200,2500
    first_tran_end = find_end_transmission(abs_f[first_tran_start+min_trans:first_tran_start+max_trans])+first_tran_start + min_trans
    print("first_transm_loc",first_tran_start,first_tran_end)

    min_trans_delay,max_trans_delay=3000,10000
    second_tran_start = find_start_transmission(abs_f[first_tran_end+min_trans_delay:first_tran_end+max_trans_delay])+first_tran_end+min_trans_delay
    print("second start loc",second_tran_start)

    #Read data between, see if we are RN16 or EPC
    reflected_data_loc = find_RN16(abs_f[first_tran_end+100:second_tran_start-100])+first_tran_end+100
    print("Reflected data loc is ",reflected_data_loc)
    rn16_test = decode_RN16(abs_f[reflected_data_loc-20:second_tran_start-20],7,0)
    data_len = len(rn16_test)

    if data_len<18 and data_len>14:
        #probably a RN16
        print("Found a RN16",rn16_test,len(rn16_test))
        #Start position is the first transmsiion
        return first_tran_start
        
    elif data_len>=100:
        #probably an epc
        print("Found a EPC",rn16_test,len(rn16_test))
        return second_tran_start
        
    else:
        #failed read
        print("Failed to read an EPC or RN16 in range ",0+relative_position,10000+relative_position,"datalen was", data_len,"LOOPING")
        return find_valid_transmission(relative_position+5000)
plt.plot(abs_f)
plt.show() 
find_valid_transmission(0)
# Find and plot transmission starts



'''
#transmission_starts = find_initial_transmissions(abs_f)
#y = np.ones(len(transmission_starts))*1.06*np.amax(abs_f)
#plt.scatter(transmission_starts,y,c='r',marker='x')

start_RN16=[]
start_EPC = []
offset = 1900 # How far after the start of a transmission the RN16 can be expected
for x in range(len(transmission_starts)-1):
    relative_start_loc = find_RN16(abs_f[transmission_starts[x]+offset:transmission_starts[x+1]])
    # If the time between transmission starts is longer than 7000, it is an EPC not RN16
    if transmission_starts[x]+7000>transmission_starts[x+1]:
        # This now doesn't fail, but will pick incorrect RN16 when no RN16 is present
        start_RN16.append(transmission_starts[x]+offset + relative_start_loc)
    else:
        start_EPC.append(transmission_starts[x]+offset + relative_start_loc)
    
    #Decode the ACK signals
    ACK = decode_RN16(abs_f[transmission_starts[x]-100:(transmission_starts[x]+offset)],9,1)
    #print("ACK IS ",ACK)
    plt.text(transmission_starts[x],1.01*np.amax(abs_f),int(''.join(map(str,ACK)),2))
#print(start_RN16)

print("Number of transmissions",len(transmission_starts))
print("Sum of RN16s + EPCs",len(start_RN16))

y_start_RN16 = np.ones(len(start_RN16))*1.06*np.amax(abs_f)
plt.scatter(start_RN16,y_start_RN16,c='b',marker='x')
#y_failed_RN16 = np.ones(len(failed_RN16))*1.02*np.amax(abs_f)
#plt.scatter(failed_RN16,y_failed_RN16,c='r',marker='o')

# Decode RN16s, add text to the plot
for start in start_RN16:
    #print("Start point is ",start)
    #print("data to decode is ", abs_f[start:(start+200)])
    RN16 = decode_RN16(abs_f[start-10:(start+1500)],7,0)
    print("RN16 value is ",RN16,len(RN16))
    plt.text(start+200,1.09*np.amax(abs_f),int(''.join(map(str,RN16)),2))
    
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
'''

