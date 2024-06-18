import matplotlib.pyplot as plt
from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from plots import plotMTFs, animate, plotModulations, plotSpectras, plotSurfaceSAR,coherence_time,plotCovariances,plotSpeckle
from parameters import parameters
import sys
if len(sys.argv)!=2:
    print("Usage:\n\t\tpython3 simulation.py parameter_file.yml")
    exit()
params=parameters(sys.argv[1])
#%% Surface Generation
from surface import surfaceGenerator
surfaceGenerator=surfaceGenerator(params)
Z=surfaceGenerator.generate()
animate(Z)

#%% Sar Imaging
from SAR_imaging import SAR_imaging
sar=SAR_imaging(surfaceGenerator, params)
sar.generate()
plotMTFs(sar)
plotModulations(sar)
plotSpectras(sar)
plotSurfaceSAR(sar)
# plotCovariances(sar)
# plotSpeckle(sar)
integral_covariance=np.trapz(np.trapz((abs(sar.orbital_velocity_mtf()))**2*sar.PSI,sar.ky[:,0],axis=0),sar.kx[0,:],axis=0)



plt.legend()
plt.show()
