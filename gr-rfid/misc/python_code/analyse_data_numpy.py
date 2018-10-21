import scipy
from os import getcwd

relative_path_to_file = '../data/Corilateme/source'


f = scipy.fromfile(open(getcwd()+'/'+relative_path_to_file), dtype=scipy.float32)
print(f.size)
f=f[36000:100000]
abs_f=abs(f[0::2]+1j*f[1::2])



import matplotlib.pyplot as plt
plt.plot(abs_f)
plt.show()

