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
steps = 100
#step size parameter
beta = np.arange(0.5, 1.0, 0.01)
alpha = 0.5
#eta = 0.9

#sum squared error
step=np.arange(0, steps, 2)
'''
sse_chioa=np.zeros(steps//2)

for i in range(0,steps):
    guess, prev=chioa(guess, prev, mask, alpha, beta)
    if(i%2==0):
        sse_chioa[i//2]=calculateSSE(guess)
        print(i)

imageio.imwrite('testpat1_chioa.jpeg', prev)
print("CHIOA done")

#initial guess using random phase info
guess = fourier_transform * np.exp(1j*2*np.pi*np.random.rand(*padded_image.shape))
#previous result
prev = None
#number of iterations
steps = 1000
#step size parameter
beta = 0.9
#sum squared error
step=np.arange(0, steps, 2)
sse_raara=np.zeros(steps//2)

for i in range(0,steps):
    guess, prev=raara(guess, prev, mask, beta)
    beta=beta+(1-beta)*(1-np.exp(-np.square(i/100)))
    if(i%2==0):
        sse_raara[i//2]=calculateSSE(guess)
        print(i)

imageio.imwrite('testpat1_raara.jpeg', prev)
print("RAARA done")

#initial guess using random phase info
guess = fourier_transform * np.exp(1j*2*np.pi*np.random.rand(*padded_image.shape))
#previous result
prev = None
#number of iterations
steps = 1000
#step size parameter
beta = 0.8
#sum squared error
step=np.arange(0, steps, 2)
sse_hioa=np.zeros(steps//2)

for i in range(0,steps):
    guess, prev=hioa(guess, prev, mask, beta)
    if(i%2==0):
        sse_hioa[i//2]=calculateSSE(guess)
        print(i)

imageio.imwrite('testpat1_hioa.jpeg', prev)
print("HIOA done")


#initial guess using random phase info
guess = fourier_transform * np.exp(1j*2*np.pi*np.random.rand(*padded_image.shape))
#previous result
prev = None
#sum squared error
sse_er=np.zeros(steps//2)

for i in range(0,steps):
    guess, prev=er(guess, prev, mask, beta)
    if(i%2==0):
        sse_er[i//2]=calculateSSE(guess)
        print(i)

imageio.imwrite('circle_er.jpeg', prev)
print("ER done")
'''

sse=np.zeros(beta.shape[0])
for i in range(50, 100, 1):
    #initial guess using random phase info
    guess = guess_
    #previous result
    prev = None
    for j in range(0,steps):
        guess, prev=chioa(guess, prev, mask, alpha, beta[i-50])
    sse[i-50]=calculateSSE(guess)
    print(i)

print("done")

fig, ax = plt.subplots()
ax.plot(beta, sse, color='red', linewidth=0.5)
ax.set(xlabel=r'$\beta$', ylabel='SSE in dB', title=r'SSE values varying $\beta$')
plt.show()
