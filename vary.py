import numpy as np
import numpy.fft as fft
import imageio
import matplotlib.pyplot as plt

i=np.arange(1000)
beta=0.5+(1-0.5)*(1-np.exp(-i/250))

fig, ax = plt.subplots()
ax.plot(i, beta, linewidth=0.5)
ax.set(xlabel='Number of Iterations (k)', ylabel=r'$\beta$', title=r'$\beta$ values as a function of the number of iterations')
plt.show()
