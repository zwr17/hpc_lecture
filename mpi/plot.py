import numpy as np
import matplotlib.pyplot as plt

file = open("output")
data = np.loadtxt(file)
plt.loglog(data[:,0],data[:,1],'o-')
plt.xlabel('N')
plt.ylabel('time')
plt.show()
