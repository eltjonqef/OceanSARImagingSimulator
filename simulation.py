import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from plots import plotMTFs, animate, plotModulations, plotSpectras
spectrum=spectrum_model.JONSWAP
spreading=spreading_model.Simple_Cosine

    

n=6
S=8
length=512
N=256
wind_speed=6
wind_direction=np.pi/4
seconds=5
timestep=0.5
fetch=25000
elfouhaily_k=0.1
spatial_resolution=2
integration_time=0.5
#%% Surface Generation
from surface import surfaceGenerator
surfaceGenerator=surfaceGenerator(spectrum, spreading, length, N, wind_speed, wind_direction, n, S, seconds, timestep, fetch, elfouhaily_k)
Z=surfaceGenerator.generate()
animate(Z)
print(f"Surface Variances {surfaceGenerator.getSurfaceVariances()}")
print(f"Spectrum Integral {surfaceGenerator.getSpectrumIntegral()}")
print(f"Slope Variances {surfaceGenerator.getSlopesVariance()}")
print(f"Slope Integral {surfaceGenerator.getSlopeIntegral()}")
print(f"Significant wave height {surfaceGenerator.getSignificantWaveHeights()}")
#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(surfaceGenerator, length, N,spectrum_model.Pierson_Moskowitz, np.deg2rad(35), wind_speed, wind_direction, fetch, spatial_resolution, integration_time)
sar.generate()
plotMTFs(sar)
plotModulations(sar)
plotSpectras(sar)
fig, (axSurface,axSurfaceSu)=plt.subplots(1,2)
axSurface.imshow(sar.surface, origin='lower')
axSurfaceSu.imshow(sar.I, origin='lower')
print(f"mtf max {np.max(sar.tilt_mtf())}")
# print(f"covariance 0 {sar.v_covariance()}")
integral_covariance=np.trapz(np.trapz((abs(sar.orbital_velocity_mtf()))**2*sar.PSI,sar.kx[0,:],axis=0),sar.ky[:,0],axis=0)
print(f"integral covariance {integral_covariance}")
fig1, (ax1)=plt.subplots(1)
x, y=np.meshgrid(np.linspace(-sar.L/2,sar.L/2,sar.N), np.linspace(-sar.L/2,sar.L/2,sar.N))

ax1.imshow(np.real(integral_covariance*np.exp(1j*sar.wavenumbers*np.sqrt(x**2+y**2))),origin='lower')
# ax1.plot((sar.v_covariance()[N//2,:]))
# ax1.plot((sar.v_covariance()[:,N//2]))
# x, y=np.meshgrid(np.linspace(0, sar.L, sar.N), np.linspace(0, sar.L, sar.N))
# ax2.contour(sar.kx, sar.ky,(abs(sar.orbital_velocity_mtf())))
# ax2.plot(0.5*np.real(np.trapz((abs(sar.orbital_velocity_mtf())**2)*(sar.PSI),sar.wavenumbers [0,:], axis=0))*np.exp(1j*sar.wavenumbers*np.sqrt(x**2+y**2)))
# ax1.plot((sar.v_covariance())[:,N//2],label="middle row")
# ax1.plot(sar.v_covariance()[:,128],label="middle column")
# ax1.set_ylim([-1,1])
print(f"theoretical {np.trapz(np.trapz(sar.dispersion_relation*sar.PSI,sar.kx[0,:], axis=0),sar.ky[:,0],axis=0)}, covaraince {sar.v_covariance()[N//2,N//2]}")

plt.legend()
#------Orbital Velocity Verification-------
# # # # fig2, (axa,axb, axOV, axOVsum)=plt.subplots(1,4)
# # # # axa.imshow(sar.surface, origin='lower')
# # # # axb.imshow(sar.hta, origin='lower')
# # # # ovColorbar=axOV.imshow(sar.u_r, origin='lower')
# # # # plt.colorbar(ovColorbar, ax=axOV)
# # # # ovSumColorbar=axOVsum.imshow(sar.u_r_sum, origin='lower')
# # # # plt.colorbar(ovSumColorbar, ax=axOVsum)
# # # # print(f"{np.var(sar.surface)} {np.var(sar.hta)} {np.var(sar.u_r)} {np.var(sar.u_r_sum)}")
# figTilt, (axSurface, axTilt, axGrad)=plt.subplots(1,3)
# axSurface.imshow(Z[0,:,:], origin='lower')
# tilt=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
# axTilt.imshow(tilt, origin='lower')
# axGrad.imshow(np.gradient(Z[0,:,:],axis=1), origin='lower')
# # plt.plot(abs(sar.velocity_bunching_mtf()))
# plt.legend()
# plt.plot(np.real(np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))[:,0],label='Tilt')
# plt.plot(sar.surface[:,0],label='Surface')
# plt.legend()
# plt.show()
# fig1, (ax1, ax2, ax3)=plt.subplots(1,3)
# ax1.imshow(Z[0,:,:], extent=[0,length,0,length], origin='lower')
# ax1.set_title("Wave Field")
# ax2.imshow(sar.image(), extent=[0,length,0,length], origin='lower')
# ax2.set_title("SAR Image")
# ax3.imshow(sar.noisy_image(), extent=[0,length,0,length], origin='lower')
# ax3.set_title("Noisy SAR Image")

# fig2, ((ax3,ax4), (ax5, ax6))=plt.subplots(2, 2)

# ax3.contour(sar.kx, sar.ky, abs(surfaceGenerator.PSI))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
# ax3.set_title("Original Spectrum")
# ax4.contour(sar.kx, sar.ky,abs(np.fft.fftshift(np.fft.fft2(sar.surface))))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
# ax4.set_title("Sea surface Spectrum")
# print(f"min {np.min(sar.I)} {np.mean(sar.I)} {np.max(sar.I)}")
# ax5.contour(sar.kx, sar.ky,abs(np.fft.fftshift(np.fft.fft2(sar.I-np.mean(sar.I)))))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
# ax5.set_title("SAR Spectrum")
# ax6.contour(sar.kx, sar.ky,abs(sar.wave_field()))#, extent=(sar.kx.min(),sar.kx.max(),sar.ky.min(),sar.ky.max()))
# ax6.set_title("Inverse SAR")
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
