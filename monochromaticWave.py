import numpy as np

class monochromaticSurfaceGenerator:
    def __init__(self, length, N, wind_direction, seconds, timestep):
        self.g=9.81
        self.dx=length/N
        self.L=length
        self.N=N
        self.wind_direction=wind_direction
        x=np.linspace(-self.L/2,self.L/2,self.N)
        y=np.linspace(-self.L/2,self.L/2,self.N)
        self.x, self.y=np.meshgrid(x, y)
        self.time=np.linspace(0,seconds, int(seconds/timestep))
        self.surface=np.zeros((self.time.size, self.N, self.N))
        self.wavenumbers=np.array([0.5])#np.logspace(-3,5,500)
        self.theta=np.array([0])#np.linspace(0,2*np.pi, 200)
        self.omega=np.sqrt(self.g*self.wavenumbers)
    
    def generate(self):
        for frame, t in enumerate(self.time):
            for i, k in enumerate(self.wavenumbers):
                for j, theta in enumerate(self.theta):
                    self.surface[frame,:,:]+=1*np.cos(k*(self.x*np.cos(theta-self.wind_direction)+self.y*np.sin(theta-self.wind_direction))-self.omega[i]*t)
        return self.surface


if __name__ == "__main__":
    from monochromaticWave import monochromaticSurfaceGenerator
    from SAR_imaging import SAR_imaging
    import matplotlib.pyplot as plt
    from omnidirectional_spectrum import spectrum_model
    from spreading_function import spreading_model
    import numpy as np
    length=128
    N=256
    seconds=5
    timestep=0.5
    wind_direction=0
    monochromatic=monochromaticSurfaceGenerator(length, N, wind_direction, seconds, timestep)
    Z=monochromatic.generate()
    fig, (ax1, ax2, ax3, ax4)=plt.subplots(1, 4)
    ax1.imshow(Z[0,:,:], extent=[0,length,0,length], origin='lower')
    ax1.set_title("Wave Field")
    sar=SAR_imaging(Z[0,:,:], length, N,spectrum_model.Pierson_Moskowitz, np.pi/6, monochromatic.wavenumbers,  monochromatic.omega, 10, 0, 25000)
    ax2.imshow(sar.NRCS(), extent=[0,length,0,length], origin='lower')
    ax2.set_title("NRCS")
    ax3.imshow(sar.image(), extent=[0,length,0,length], origin='lower')
    ax3.set_title("SAR Image")

    thetas=np.linspace(0, np.pi/2, 100)
    sigma0=[(sar.average_NRCS(theta)) for theta in thetas]
    ax4.plot(np.degrees(thetas), 10*np.log10(sigma0))
    ax4.set_ylim(-30,60)
    ax4.set_title("sigma 0")
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