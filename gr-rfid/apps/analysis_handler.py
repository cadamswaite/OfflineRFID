import subprocess

echo_cmd = ['sudo', 'GR_SCHEDULER=STS', 'nice', '-20', 'python', '/home/carl/git/Distributed-Gen2-UHF-RFID-Reader/gr-rfid/apps/analysis_reader.py']
proc = subprocess.Popen(echo_cmd, stdout=subprocess.PIPE)
output = proc.communicate()[0]
output = output.split(',')

if output[-1][:2]=='Ex':
    print("Full run time")
else:
    print("Run cut short. Increase the delay time in analysis_reader")


#TODO implement this efficiently using numpy
read_string = ''
for each_read in output:
    if each_read[:2]=='ID':
        read_string+='1'
    else:
        read_string+='0'

#print(read_string)

bin_size = 50
averages_list = []
for x in range(len(read_string)-bin_size):
    current_sum=0
    for y in range(bin_size):
        current_sum+=int(read_string[x+y])
    average = current_sum/float(bin_size)
    averages_list.append(average)    
    
import matplotlib.pyplot as plt
plt.plot(averages_list)
plt.ylabel('Average Error rate vs read number')
plt.show()


