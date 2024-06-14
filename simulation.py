import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from plots import plotMTFs, animate, plotModulations, plotSpectras, plotSurfaceSAR,coherence_time
from parameters import parameters
coherence_time()
params=parameters()
#%% Surface Generation
from surface import surfaceGenerator
surfaceGenerator=surfaceGenerator(params)
Z=surfaceGenerator.generate()
animate(Z)
# print(f"Surface Variances {surfaceGenerator.getSurfaceVariances()}")
# print(f"Spectrum Integral {surfaceGenerator.getSpectrumIntegral()}")
# print(f"Slope Variances {surfaceGenerator.getSlopesVariance()}")
# print(f"Slope Integral {surfaceGenerator.getSlopeIntegral()}")
# print(f"Significant wave height {surfaceGenerator.getSignificantWaveHeights()}")
# print(f"max {np.max(Z[0,:,:])}")
#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(surfaceGenerator, params)
sar.generate()
plotMTFs(sar)
plotModulations(sar)
plotSpectras(sar)
plotSurfaceSAR(sar)
integral_covariance=np.trapz(np.trapz((abs(sar.orbital_velocity_mtf()))**2*sar.PSI,sar.ky[:,0],axis=0),sar.kx[0,:],axis=0)



plt.legend()
#------Orbital Velocity Verification-------
# fig2, (axa,axb,axOV, axOVsum)=plt.subplots(1,4)
# surfaceColorbar=axa.imshow(sar.surface, origin='lower')
# plt.colorbar(surfaceColorbar, ax=axa)
# etaColorbar=axb.imshow(sar.hta, origin='lower')
# plt.colorbar(etaColorbar, ax=axb)
# ovColorbar=axOV.imshow(sar.u_r, origin='lower')
# plt.colorbar(ovColorbar, ax=axOV)
# ovSumColorbar=axOVsum.imshow(sar.u_r_sum, origin='lower')
# plt.colorbar(ovSumColorbar, ax=axOVsum)
# print(f"{np.var(sar.surface)} {np.var(sar.hta)} {np.var(sar.u_r)} {np.var(sar.u_r_sum)}")
plt.show()
