import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
spectrum=spectrum_model.Pierson_Moskowitz
spreading=spreading_model.Simple_Cosine
n=6
S=8
length=256
N=512
wind_speed=10
wind_direction=np.pi/6
seconds=5
timestep=0.5
fetch=25000
elfouhaily_k=0.1
#%% Surface Generation
from surface import surfaceGenerator
surfaceGenerator=surfaceGenerator(spectrum, spreading, length, N, wind_speed, wind_direction, n, S, seconds, timestep, fetch, elfouhaily_k)
Z=surfaceGenerator.generate()
print(f"Surface Variances {surfaceGenerator.getSurfaceVariances()}")
print(f"Spectrum Integral {surfaceGenerator.getSpectrumIntegral()}")
print(f"Slope Variances {surfaceGenerator.getSlopesVariance()}")
print(f"Slope Integral {surfaceGenerator.getSlopeIntegral()}")
print(f"Significant wave height {surfaceGenerator.getSignificantWaveHeights()}")
#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(Z[0,:,:], spectrum, np.pi/3, surfaceGenerator.k,  surfaceGenerator.omega, wind_speed, wind_direction, fetch) 

fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8))=plt.subplots(2, 4)
ax1.imshow(sar.surface,extent=[0,length,0,length], origin='lower')
ax1.set_title("surface")
sar_image=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.modulation_transfer_function()*np.fft.fftshift(np.fft.fft2(sar.surface)))))#(2*np.fft.ifft2((sar.tilt())*(np.fft.fft2(sar.surface))))
ax2.imshow(sar_image,extent=[0,length,0,length], origin='lower')
ax2.set_title("mtf")

ax3.imshow(sar.orbital_velocity(),extent=[0,length,0,length], origin='lower')
ax3.set_title("orbital velocity")


sar_image=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))#(2*np.fft.ifft2((sar.tilt())*(np.fft.fft2(sar.surface))))
ax5.imshow(sar_image,extent=[0,length,0,length], origin='lower')
ax5.set_title("tilt")

sar_image=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.hydrodynamic_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))#(2*np.fft.ifft2((sar.tilt())*(np.fft.fft2(sar.surface))))
ax6.imshow(sar_image,extent=[0,length,0,length], origin='lower')
ax6.set_title("hydrodynamic")

sar_image=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.range_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))#(2*np.fft.ifft2((sar.tilt())*(np.fft.fft2(sar.surface))))
ax7.imshow(sar_image,extent=[0,length,0,length], origin='lower')
ax7.set_title("range bunching")

sar_image=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.velocity_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))#(2*np.fft.ifft2((sar.tilt())*(np.fft.fft2(sar.surface))))
ax8.imshow(sar_image,extent=[0,length,0,length], origin='lower')
ax8.set_title("velocity bunching")

plt.show()


# ax3.imshow(np.gradient(sar.surface, length/N,axis=1),extent=[0,length,0,length], origin='lower')
# tilt=sar.tilt()#4.5*surfaceGenerator.omega[0,:]*(surfaceGenerator.k[0,:]*np.sin(theta))**2*(surfaceGenerator.omega[0,:]-0.01*1j)/(np.abs(surfaceGenerator.k[0,:])*(surfaceGenerator.omega[0,:]**2+0.1**2))
# hydrodynamic=sar.hydrodynamic()
# fig, (ax1, ax2)=plt.subplots(1, 2)
# ax1.plot(np.linspace(0,360,N),np.abs(tilt), label="tilt")
# # ax1.plot(np.linspace(0,360,N),np.abs(hydrodynamic), label="hydrodynamic")
# ax1.set_xlim(0, 360)
# ax1.set_xlabel('phi')
# ax1.set_title('Magnitude MTF')
# ax1.grid(True)
# # ax1.legend()

# # ax2.subplot(1, 2)
# ax2.plot(np.linspace(0,360,N),np.degrees(np.angle(tilt)), label="tilt")
# # ax2.plot(np.linspace(0,360,N),np.degrees(np.angle(hydrodynamic)), label="hydrodynamic")
# ax2.set_xlim(0, 360)
# ax2.set_xlabel('phi')
# ax2.set_title('Phase MTF')
# ax2.grid(True)
# # ax2.tight_layout()

# fig, (ax1, ax2)=plt.subplots(1, 2)
# # print(kx[0,:])
# # print(ky[:,0])
# ax1.contour(np.abs(sar.tilt()))
# ax2.contour(np.angle(sar.tilt()))
# ax1.contour(surfaceGenerator.k[0,:]*np.cos(np.linspace(0, 2*np.pi,N)),surfaceGenerator.k[:,0]*np.sin(np.linspace(0, 2*np.pi,N)),abs(sar.tilt()),colors='w')
# ax2.contour(kx[0,:],ky[:,0],abs(sar.tilt()),colors='w')
# ax1.imshow(abs(sar.tilt()))
# ax2.imshow(np.degrees(np.angle(sar.tilt())))
