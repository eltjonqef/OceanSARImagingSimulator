import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
spectrum=spectrum_model.Pierson_Moskowitz
spreading=spreading_model.Simple_Cosine
n=6
S=8
length=512
N=256
wind_speed=10
wind_direction=np.pi/6
seconds=5
timestep=0.5
fetch=25000
elfouhaily_k=0.1
spatial_resolution=5
integration_time=0.66
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
sar=SAR_imaging(Z[0,:,:], length, N,spectrum_model.Pierson_Moskowitz, np.pi/6, surfaceGenerator.k,  surfaceGenerator.omega, wind_speed, wind_direction, fetch, spatial_resolution, integration_time)

fig1, (ax1, ax2, ax3)=plt.subplots(1,3)
ax1.imshow(Z[0,:,:], extent=[0,length,0,length], origin='lower')
ax1.set_title("Wave Field")
ax2.imshow(sar.image(), extent=[0,length,0,length], origin='lower')
ax2.set_title("SAR Image")
ax3.imshow(sar.noisy_image(), extent=[0,length,0,length], origin='lower')
ax3.set_title("Noisy SAR Image")

fig2, ((ax3,ax4), (ax5, ax6))=plt.subplots(2, 2)

ax3.contour(abs(np.fft.fftshift(surfaceGenerator.PSI)), extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
ax3.set_title("Original Spectrum")
ax4.contour(abs(np.fft.fftshift(np.fft.fft2(sar.surface))), extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
ax4.set_title("Sea surface Spectrum")
ax5.contour(abs(np.fft.fftshift(np.fft.fft2(sar.I))), extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
ax5.set_title("SAR Spectrum")
ax6.contour(abs(sar.wave_field()), extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
ax6.set_title("Inverse SAR")
# ax4.plot(Z[0,0,:], label="image")
# ax4.plot(sar.NRCS()[0,:], label="NRCS")
# tilt=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
# ax4.plot(tilt[0,:], label="tilt")
# hydrodynamic=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.hydrodynamic_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
# ax4.plot(hydrodynamic[0,:], label="hydrodynamic")
# rb=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.range_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
# ax4.plot(rb[0,:], label="range bunching")
# ax4.set_title("MTF")
# ax4.legend()
plt.show()
