import numpy as np
import numpy.fft as fft
import imageio
import matplotlib.pyplot as plt

#read in input image
#m=4
input_image = imageio.imread('cameraman.png', as_gray=True)
#input_image = np.kron(input_img, np.ones((m,m)))
x_len = input_image.shape[0]
y_len = input_image.shape[1]
padded_image = np.pad(input_image, ((x_len, x_len), (y_len, y_len)), 'constant')
fourier_transform = np.abs(fft.fft2(padded_image))
normalization=np.sum(np.square(fourier_transform))
mask = np.pad(np.ones((x_len+2,y_len+2)), ((x_len-1,x_len-1), (y_len-1,y_len-1)), 'constant')

def hioa(guess, prev, mask, beta):
    #apply fourier domain constraints
    updated_guess = fourier_transform * np.exp(1j * np.angle(guess)) 
    inv = np.real(fft.ifft2(updated_guess))
    if prev is None:
        prev = inv        
    #apply real-space constraints
    temp = inv
    gamma=np.logical_or(np.logical_and(inv<0, mask), np.logical_not(mask))
    inv[gamma]=prev[gamma] - beta * inv[gamma]
    return fft.fft2(inv), temp

def chioa(guess, prev, mask, alpha, beta):
    #apply fourier domain constraints
    updated_guess = fourier_transform * np.exp(1j * np.angle(guess)) 
    inv = np.real(fft.ifft2(updated_guess))
    if prev is None:
        prev = inv        
    #apply real-space constraints
    temp = inv
    gamma1=np.logical_or(np.logical_and(np.logical_and(inv>=0, mask), np.logical_and(inv<=alpha*prev, mask)), np.logical_not(mask))
    gamma2=np.logical_or(np.logical_and(inv<0, mask), np.logical_not(mask))
    inv[gamma1]=prev[gamma1] - (1-alpha)/alpha * inv[gamma1]
    inv[gamma2]=prev[gamma2] - beta * inv[gamma2]
    return fft.fft2(inv), temp

def er(guess, prev, mask, beta):
    #apply fourier domain constraints
    updated_guess = fourier_transform * np.exp(1j * np.angle(guess)) 
    inv = np.real(fft.ifft2(updated_guess))
    if prev is None:
        prev = inv        
    #apply real-space constraints
    temp = inv
    gamma=np.logical_or(np.logical_and(inv<0, mask), np.logical_not(mask))
    inv[gamma]=0
    return fft.fft2(inv), temp

def raara(guess, prev, mask, beta):
    #apply fourier domain constraints
    updated_guess = fourier_transform * np.exp(1j * np.angle(guess)) 
    inv = np.real(fft.ifft2(updated_guess))
    if prev is None:
        prev = inv        
    #apply real-space constraints
    temp = inv
    gamma=np.logical_or(np.logical_and(inv<0, mask), np.logical_not(mask))
    inv[gamma]=beta*prev[gamma] + (1-2*beta) * inv[gamma]
    return fft.fft2(inv), temp

def calculateSSE(sample):
    error=np.sum(np.square(np.abs(sample)-fourier_transform))/normalization
    return 10*np.log(error)	


#initial guess using random phase info
guess_ = fourier_transform * np.exp(1j*2*np.pi*np.random.rand(*padded_image.shape))
#previous result
prev = None
#number of iterations
steps = 1000
iterations = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80])
#step size parameter
beta = 0.8
alpha = 0.6
#eta = 0.9

#sum squared error
step=np.arange(0, steps, 2)
sse=np.zeros((iterations.shape[0], steps//2))

for i in range(0, iterations.shape[0], 1):
    #initial guess using random phase info
    guess = guess_
    #previous result
    prev = None

    for j in range(0,steps):
        if(j%90<=iterations[i] or j>600):
            guess, prev=chioa(guess, prev, mask, alpha, beta)
        else:
            guess, prev=hioa(guess, prev, mask, beta)
        if(j%2==0):
            sse[i][j//2]=calculateSSE(guess)
    print(i)

fig, ax = plt.subplots()
cmap=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'lime', 'lawngreen', 'blue', 'dodgerblue']
for i in range(0, iterations.shape[0], 1):
    ax.plot(step, sse[i,:], color=cmap[i], linewidth=0.6, label=f'({iterations[i]+1}, {90-iterations[i]-1})')
ax.set(xlabel='Number of Iterations', ylabel='SSE in dB', title='SSE values using CHIOA/HIOA with different cycle lengths')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(iterations, sse[:,-1], color='red', linewidth=0.5)
ax.set(xlabel='Number of Iterations of CHIOA (k)', ylabel='SSE in dB', title='SSE values using CHIOA/HIOA varying the number of iterations of CHIOA')
plt.show()
