#https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-4-7-786
#https://arxiv.org/abs/1709.01071
#https://www.spiedigitallibrary.org/journals/optical-engineering/volume-56/issue-9/094103/Sparse-superresolution-phase-retrieval-from-phase-coded-noisy-intensity-patterns/10.1117/1.OE.56.9.094103.full
#https://www.mdpi.com/2076-3417/8/5/719
#https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10677/106771B/Multiwavelength-surface-contouring-from-phase-coded-diffraction-patterns/10.1117/12.2306127.short
#https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10335/1033509/Computational-wavelength-resolution-for-in-line-lensless-holography--phase/10.1117/12.2269327.short
#https://www.spiedigitallibrary.org/journals/optical-engineering/volume-57/issue-8/085105/Multiwavelength-surface-contouring-from-phase-coded-noisy-diffraction-patterns/10.1117/1.OE.57.8.085105.full
#https://ieeexplore.ieee.org/abstract/document/8553264
#https://webpages.tuni.fi/lasip/DDT/pdfs/Sparse%20phase%20retrieval%20from%20noisy%20data%20December_28_2015.pdf

import numpy as np
import numpy.fft as fft
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm

#read in input image
input_image = imageio.imread('cameraman.png', as_gray=True)
x_len = input_image.shape[0]
y_len = input_image.shape[1]
padded_image = np.pad(input_image, ((x_len, x_len), (y_len, y_len)), 'constant')
fourier_transform = fft.fft2(padded_image)

'''
output_image_er = imageio.imread('circle_er.jpeg', as_gray=True)
fourier_transform_output_er = fft.fft2(output_image_er)
delta_phi_er=np.angle(fourier_transform)-np.angle(fourier_transform_output_er)

output_image_hioa = imageio.imread('circle_hioa.jpeg', as_gray=True)
fourier_transform_output_hioa = fft.fft2(output_image_hioa)
delta_phi_hioa=np.angle(fourier_transform)-np.angle(fourier_transform_output_hioa)
'''

output_image_mixed = imageio.imread('image_mixed.jpeg', as_gray=True)
fourier_transform_output_mixed = fft.fft2(output_image_mixed)
delta_phi_mixed=np.angle(fourier_transform)-np.angle(fourier_transform_output_mixed)

output_image_mixed_ = imageio.imread('image_mixed1_1.jpeg', as_gray=True)
fourier_transform_output_mixed_ = fft.fft2(output_image_mixed_)
delta_phi_mixed_=np.angle(fourier_transform)-np.angle(fourier_transform_output_mixed_)


#fig, axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
x=np.arange(0, 3*x_len, 1)
y=np.arange(0, 3*y_len, 1)
X, Y=np.meshgrid(x, y)
# Plot the surface.
'''
surf_er = axs[0].plot_surface(X, Y, delta_phi_er, cmap=cm.nipy_spectral, linewidth=0, antialiased=False)
axs[0].set_xlabel('Pixels')
axs[0].set_ylabel('Pixels')
axs[0].set_zlabel('$\Delta \Phi (in rd)$', rotation = 90)
axs[0].title.set_text('ERA')
surf_hioa = axs[1].plot_surface(X, Y, delta_phi_hioa, cmap=cm.nipy_spectral, linewidth=0, antialiased=False)
axs[1].set_xlabel('Pixels')
axs[1].set_ylabel('Pixels')
axs[1].set_zlabel('$\Delta \Phi (in rd)$', rotation = 90)
axs[1].title.set_text('HIOA')
'''
surf_mixed = axs[0].plot_surface(X, Y, delta_phi_mixed, cmap=cm.nipy_spectral, linewidth=0, antialiased=False)
axs[0].set_xlabel('Pixels')
axs[0].set_ylabel('Pixels')
axs[0].set_zlabel('$\Delta \Phi (in rd)$', rotation = 90)
axs[0].title.set_text('12 ER/HIOA(1,49) + 400 ERA')

surf_mixed_ = axs[1].plot_surface(X, Y, delta_phi_mixed_, cmap=cm.nipy_spectral, linewidth=0, antialiased=False)
axs[1].set_xlabel('Pixels')
axs[1].set_ylabel('Pixels')
axs[1].set_zlabel('$\Delta \Phi (in rd)$', rotation = 90)
axs[1].title.set_text('12 CHIOA/HIOA(1,49) + 160 CHIOA')
plt.show()
