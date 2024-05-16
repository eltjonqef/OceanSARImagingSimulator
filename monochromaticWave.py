import numpy as np
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model
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
        self.KX=np.array([0.5])
        self.KY=np.array([0.5])
        self.K=np.sqrt(self.KX**2+self.KY**2)#np.array([0.5])#np.logspace(-3,5,500)
        self.theta=np.array([0])#np.linspace(0,2*np.pi, 200)
        self.omega=np.sqrt(self.g*self.K)
        self.omnidirectional_spectrum=omnidirectional_spectrum(spectrum=spectrum_model.Pierson_Moskowitz,k=self.K,v=10,F=None, good_k=None)
        # self.omnidirectional_spectrum.plot()
        self.spreading_function=spreading_function(function=spreading_function.Simple_Cosine, theta=self.theta, n=2, S=None, F=None, k=None, v=10, good_k=None)
        # self.spreading_function.plot()
        S=self.omnidirectional_spectrum.getSpectrum()
        D=self.spreading_function.getSpread()
        self.PSI=np.sqrt(S)#*D
        self.random_phase=1
        self.wave_coeffs=1
        # self.KY=self.KY#*np.sin(np.linspace(0,2*np.pi,self.N))
        # self.KX=self.KY*np.cos(np.linspace(0,2*np.pi,self.N))

        print(self.PSI)

    
    def generate(self):
        for frame, t in enumerate(self.time):
            for i, k in enumerate(self.K):
                for j, theta in enumerate(self.theta):
                    self.surface[frame,:,:]+=1*np.cos(k*(self.x*np.cos(self.wind_direction)+self.y*np.sin(self.wind_direction))-self.omega[i]*t)
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
    wind_speed=6
    wind_direction=0#np.pi/2
    monochromatic=monochromaticSurfaceGenerator(length, N, wind_direction, seconds, timestep)
    Z=monochromatic.generate()
    # plt.imshow(Z[0,:,:],origin='lower')
    # plt.show()
    spatial_resolution=5
    integration_time=0.66
    sar=SAR_imaging(monochromatic, length, N,spectrum_model.Pierson_Moskowitz, np.pi/6, wind_speed, 0, 25000, spatial_resolution, integration_time)

    # plt.plot(sar.surface[:,0], label="Surface")
    plt.plot(np.real(np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))[:,0], label="Tilt")
    # plt.colorbar()
    plt.legend()
    fig, (ax4)=plt.subplots(1)
    ax4.plot(Z[0,0,:], label="Z")
    # ax4.plot(sar.NRCS()[0,:], label="NRCS")
    tilt=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    ax4.plot(tilt[0,:], label="tilt")
    plt.show()
    fig1, (ax1, ax2, ax3)=plt.subplots(1,3)
    ax1.imshow(Z[0,:,:], extent=[0,length,0,length], origin='lower')
    ax1.set_title("Wave Field")
    ax2.imshow(np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.tilt_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface))))), extent=[0,length,0,length], origin='lower')
    ax2.set_title("SAR Image")
    plt.show()
    ax3.imshow(sar.noisy_image(), extent=[0,length,0,length], origin='lower')
    ax3.set_title("Noisy SAR Image")
    fig1, (orbitalLine,orbitalImage)=plt.subplots(1,2)
    orbitalLine.plot(sar.surface[:,5], label='Surface')
    orbitalLine.plot(sar.orbital_velocity()[:,5], label='Orbital Velocity')
    orbitalLine.legend()
    orbitalImage.imshow(sar.orbital_velocity(), origin='lower')
    plt.show()
    # thetas=np.linspace(0, np.pi/2, 100)
    # sigma0=[(sar.average_NRCS(theta)) for theta in thetas]
    # ax4.plot(np.degrees(thetas), 10*np.log10(sigma0))
    # ax4.set_ylim(-30,60)
    # ax4.set_title("sigma 0")
    # # hydrodynamic=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.hydrodynamic_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    # # ax4.plot(hydrodynamic[0,:], label="hydrodynamic")
    # # rb=np.real(2*np.fft.ifft2(np.fft.ifftshift(sar.range_bunching_mtf()*np.fft.fftshift(np.fft.fft2(sar.surface)))))
    # # ax4.plot(rb[0,:], label="range bunching")
    # ax4.plot(sar.image()[0,:], label="Intensity")
    # ax4.set_title("MTF")
    # ax4.legend()

    fig1, (ax1, ax2, ax3)=plt.subplots(1,3)
    ax1.imshow(sar.surface, extent=[0,length,0,length], origin='lower')
    ax1.set_title("Wave Field")
    ax2.imshow(sar.image(), extent=[0,length,0,length], origin='lower')
    ax2.set_title("SAR Image")
    ax3.imshow(sar.noisy_image(), extent=[0,length,0,length], origin='lower')
    ax3.set_title("Noisy SAR Image")
    fig2, ((ax5,ax6), (ax7, ax8))=plt.subplots(2, 2)
    aaa=abs(np.fft.fftshift(monochromatic.PSI))
    # print(aaa)
    ax5.plot(abs(np.fft.fftshift(monochromatic.PSI)))
    ax5.set_title("Original Spectrum")
    ax6.plot(abs(np.fft.fftshift(np.fft.fft2(sar.surface))))
    ax6.set_title("Sea surface Spectrum")
    ax7.plot(abs(np.fft.fftshift(np.fft.fft2(sar.I))))
    ax7.set_title("SAR Spectrum")
    ax8.plot(abs(sar.wave_field()))
    ax8.set_title("Inverse SAR")
    # fig, (ax5)=plt.subplots(1)
    # speeds=np.linspace(1, 20, 9)
    # cts=[sar.coherence_time(speed) for speed in speeds]
    # ax5.plot(speeds, cts)
    plt.show()