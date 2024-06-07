import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from plots import plotMTFs, animate, plotModulations, plotSpectras, plotSurfaceSAR
spectrum=spectrum_model.Elfouhaily
spreading=spreading_model.Longuet_Higgins

    

n=6
S=12
length=256
N=256
wind_speed=5
wind_direction=np.pi/2
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
print(f"max {np.max(Z[0,:,:])}")
#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(surfaceGenerator, length, N,spectrum_model.Pierson_Moskowitz, np.deg2rad(35), wind_speed, wind_direction, fetch, spatial_resolution, integration_time)
sar.generate()
plotMTFs(sar)
plotModulations(sar)
plotSpectras(sar)
plotSurfaceSAR(sar)
integral_covariance=np.trapz(np.trapz((abs(sar.orbital_velocity_mtf()))**2*sar.PSI,sar.kx[0,:],axis=0),sar.ky[:,0],axis=0)



plt.legend()
#------Orbital Velocity Verification-------
# # # # fig2, (axa,axb,axOV, axOVsum)=plt.subplots(1,4)
# # # # surfaceColorbar=axa.imshow(sar.surface, origin='lower')
# # # # plt.colorbar(surfaceColorbar, ax=axa)
# # # # etaColorbar=axb.imshow(sar.hta, origin='lower')
# # # # plt.colorbar(etaColorbar, ax=axb)
# # # # ovColorbar=axOV.imshow(sar.u_r, origin='lower')
# # # # plt.colorbar(ovColorbar, ax=axOV)
# # # # ovSumColorbar=axOVsum.imshow(sar.u_r_sum, origin='lower')
# # # # plt.colorbar(ovSumColorbar, ax=axOVsum)
# # # # print(f"{np.var(sar.surface)} {np.var(sar.hta)} {np.var(sar.u_r)} {np.var(sar.u_r_sum)}")
plt.show()
