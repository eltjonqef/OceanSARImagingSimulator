import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from enum import Enum
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from scipy import special
from parameters import parameters
from surface import surfaceGenerator
class Polarization(Enum):
    Vertical=0
    Horizontal=1
    Both=2

class Band(Enum):
    X=0
    C=1
    L=2

class SAR_imaging:
    def __init__(self, surfaceGenerator:surfaceGenerator, params:parameters):
        self.random_phase=surfaceGenerator.random_phase
        self.PSI=surfaceGenerator.PSI
        self.wave_coeffs=surfaceGenerator.wave_coeffs
        self.polarization=Polarization.Vertical
        self.H=800000
        self.V=7000
        self.R=self.H/np.cos(params.incidence_angle)
        self.beta=self.R/self.V
        if params.beta!=None:
            self.beta=params.beta
        print(self.beta)
        self.L=params.length
        self.N=params.N
        self.dx=params.length/params.N
        self.surface=surfaceGenerator.surface[0,:,:]
        self.wind_speed=params.wind_speed
        self.wind_direction=params.wind_direction
        self.fetch=params.fetch
        self.incidence_angle=params.incidence_angle
        self.azimuth_resolution=params.azimuth_resolution
        self.range_resolution=params.range_resolution
        self.spatial_resolution=self.azimuth_resolution*self.range_resolution
        self.ky=surfaceGenerator.KY#*np.sin(np.pi/2)#*np.sin(np.linspace(0,2*np.pi, self.N))#wavenumbers*np.sin((np.linspace(0, 2*np.pi,(N))))
        self.kx=surfaceGenerator.KX#*np.cos(np.pi/2)#np.cos(np.linspace(0,2*np.pi, self.N))#wavenumbers*np.cos((np.linspace(0, 2*np.pi,(N))))
        self.wavenumbers=surfaceGenerator.K
        self.theta=np.linspace(0, 2*np.pi, (params.N))
        self.spectrum=params.spectrum
        self.dispersion_relation=surfaceGenerator.omega
        self.hydrodynamic_relaxation_rate=0.5
        self.light_speed=299792458
        self.frequency=params.frequency
        self.wavelength=self.light_speed/self.frequency
        self.integration_time=(self.wavelength*self.beta)/(2*self.azimuth_resolution)#integration_time
        self.k_e=2*np.pi/(self.light_speed/self.frequency)
        self.band=Band.C
        self.I=np.zeros((self.N,self.N))
        if self.band==Band.X:
            self.dielectric_constant=49-35.5j
        elif self.band==Band.C:
            self.dielectric_constant=60-36j
        elif self.band==Band.L:
            self.dielectric_constant=72-59j
        self.Ks=surfaceGenerator.Ks

    def generate(self):
        self.NRCS()
        self.orbital_velocity()
        # self.orbital_velocity_sum()
        self.image()
        self.v_covariance()
        self.Rv_covariance()
        self.R_covariance()

    def image(self):
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        Ur=self.u_r
        print(f"Variance of Obrital Velocities {np.var(Ur)}")
        # print(f"rho0 {self.v_covariance(0)[0,0]} max {np.max(self.v_covariance(0))}")
        # print(f"azimiuhth resolution {self.azimuth_resolution()}")
        pa=self.degraded_azimuthal_resolution()
        print(f"degraged resolution {pa}")
        # print(f"rho a {pa}")
        for i in range(self.N):
            for j in range(self.N):
                self.I[i, j]=np.pi*self.integration_time**2*self.azimuth_resolution/2*np.trapz(self.sigma[i,:]/pa*np.exp(-(np.pi/pa)**2*((x[i,j]-x[i,:]-self.beta*Ur[i,:]))**2), dx=self.dx*self.dx)
                # print(f"{i},{j} {self.I[i,j]}")
        # print(f"sigma {self.I}")
    
    def average_NRCS(self, theta=None):
        if theta is None:
            theta=self.incidence_angle
        k_brag=2*self.k_e*np.sin(theta)
        wavenumbers=np.logspace(-3,5,500)
        pos=np.abs(wavenumbers-k_brag).argmin()
        wavenumbers[pos]=k_brag
        PSI=omnidirectional_spectrum(spectrum=self.spectrum,k=wavenumbers,v=self.wind_speed,F=self.fetch).getSpectrum()[pos]
        sn=0#np.gradient(self.surface,axis=0)
        sp=0#np.gradient(self.surface,axis=1)
        theta_l=np.arccos(np.cos(theta-sp)*np.cos(sn))
        return 8*np.pi*self.k_e**4*np.cos(theta_l)**4*PSI*np.abs(self.complex_scattering())**2
    
    def NRCS(self):
        self.sigma=self.average_NRCS()*(1+np.real(2*ifft2(ifftshift(self.RAR_MTF()*fftshift(fft2(self.surface))+np.conj(self.RAR_MTF()*fftshift(fft2(self.surface)))))))

    def complex_scattering(self):
        if self.polarization==Polarization.Vertical:
            return self.dielectric_constant**2*(1+np.sin(self.incidence_angle)**2)/(self.dielectric_constant*np.cos(self.incidence_angle)+np.sqrt(self.dielectric_constant))**2
        elif self.polarization==Polarization.Horizontal:
            return self.dielectric_constant/(np.cos(self.incidence_angle)+np.sqrt(self.dielectric_constant))**2
        else:
            return None

    def tilt_mtf(self):
        return -(4/np.tan(self.incidence_angle))/(1+np.sin(self.incidence_angle)**2)*(1j*self.ky)
    
    def hydrodynamic_mtf(self):
        return 4.5*self.dispersion_relation*(self.ky)**2*(self.dispersion_relation-self.hydrodynamic_relaxation_rate*1j)/(np.abs(self.wavenumbers)*(self.dispersion_relation**2+self.hydrodynamic_relaxation_rate**2))
    
    def range_bunching_mtf(self):
        return -1j*self.ky/np.tan(self.incidence_angle)

    def velocity_bunching_mtf(self):
        return -1j*self.beta*self.kx*self.orbital_velocity_mtf()#*self.dispersion_relation*(np.sin(self.incidence_angle)*np.cos(self.wind_direction)+1j*np.cos(self.incidence_angle))
    
    def orbital_velocity_mtf(self, negative=False):
        if negative:
            return -self.dispersion_relation*(self.kx/-self.wavenumbers*np.sin(self.incidence_angle)*1j+np.cos(self.incidence_angle))
        return -self.dispersion_relation*(self.ky/self.wavenumbers*np.sin(self.incidence_angle)*1j+np.cos(self.incidence_angle))
    
    def orbital_velocity(self):
        self.u_r=np.real(ifft2(ifftshift(self.orbital_velocity_mtf()*fftshift(fft2(self.surface)))))

    def orbital_velocity_sum(self):
        x, y=np.meshgrid(np.linspace(-self.L/2,self.L/2,self.N), np.linspace(-self.L/2,self.L/2,self.N))
        g=9.81
        u_x=np.zeros((self.N,self.N))
        u_y=np.zeros((self.N,self.N))
        u_z=np.zeros((self.N,self.N))
        self.hta=np.zeros((self.N,self.N))

        print(self.wave_coeffs.shape)
        # Check shape and dtype of self.wave_coeffs
        # wave_coeffs=np.abs(self.wave_coeffs)/self.N**2
        # aaa=wave_coeffs[]
        #np.cos(k*(self.x*np.cos(self.wind_direction)+self.y*np.sin(self.wind_direction))-self.omega[i]*t)
        for i in range(self.N):
            for j in range(self.N):
                self.hta+=self.wave_coeffs[i,j]*np.cos(x*self.kx[i,j]+y*self.ky[i,j]+self.random_phase[i,j])
                u_x+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.cos(self.wind_direction)
                u_y+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.sin(self.wind_direction)
                u_z+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.sin(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])
            print(i)
        u_x*=g
        u_y*=g
        u_z*=g

        self.u_r_sum=u_z*np.cos(self.incidence_angle)-np.sin(self.incidence_angle)*(u_x*np.sin(self.wind_direction)+u_y*np.cos(self.wind_direction))

    def mean_orbital_velocity(self):
        B_f=2/(self.kx*self.dx)*np.sin(self.kx*self.dx/2)*2/(self.ky*self.dx)*np.sin(self.ky*self.dx/2)*2/(self.dispersion_relation*self.integration_time)*np.sin(self.dispersion_relation*self.integration_time/2)
        return np.real(ifft2(ifftshift(fftshift(fft2(self.u_r))*B_f)))


    
    def orbital_acceleration_mtf(self):
        return self.dispersion_relation**2*(self.ky/self.wavenumbers*np.sin(self.incidence_angle)+1j*np.cos(self.incidence_angle))

    def RAR_MTF(self):
        return self.tilt_mtf()+self.hydrodynamic_mtf()+self.range_bunching_mtf()
    
    def SAR_MTF(self):
        return self.RAR_MTF()+self.velocity_bunching_mtf()#*0.01
    

    def coherence_time(self):
        wind_speed_19_5=self.wind_speed*(19.5/10)**(1/7)
        return 3*self.wavelength/wind_speed_19_5*special.erf(2.7*self.spatial_resolution/wind_speed_19_5**2)**(-1/2)
    
    # def azimuth_resolution(self):
    #     return 5
    #     return (self.wavelength*self.beta)/(2*self.integration_time)

    def degraded_azimuthal_resolution(self):
        # print(np.sqrt(1+self.integration_time**2/self.coherence_time()**2))
        # print(self.integration_time)
        # print(self.coherence_time())
        print(f"integration time {self.integration_time}")
        print(f"coherence time {self.coherence_time()}")
        # return 2
        return self.azimuth_resolution*np.sqrt(1+(self.integration_time/self.coherence_time())**2)
    
    def noisy_image(self):
        noise=np.random.exponential(scale=1,size=(self.N, self.N))
        return self.I*noise
    
    def v_covariance(self):
        self.f_v=np.real(ifft2(ifftshift(abs(self.orbital_velocity_mtf())**2*self.PSI*(2*np.pi/self.dx)**2)))

    def R_covariance(self):
        self.f_r=0.5*np.real(ifft2(ifftshift(abs(self.RAR_MTF())**2*self.PSI*(2*np.pi/self.dx)**2+abs(np.rot90(self.RAR_MTF(),2))**2*np.rot90(self.PSI*(2*np.pi/self.dx)**2,2))))
        # return 1/2*np.trapz(np.trapz((abs(self.RAR_MTF())**2*self.PSI+np.conj(abs(self.RAR_MTF())**2*self.PSI))*np.exp(1j*self.wavenumbers*x),self.wavenumbers[0,:], axis=0),self.wavenumbers[:,0])

    def Rv_covariance(self):
        self.f_rv=0.5*np.real(ifft2(ifftshift(self.RAR_MTF()*np.conj(self.orbital_velocity_mtf())*self.PSI*(2*np.pi/self.dx)**2+np.conj(np.rot90(self.RAR_MTF(),2))*np.rot90(self.orbital_velocity_mtf(),2)*np.rot90(self.PSI*(2*np.pi/self.dx)**2,2))))
        # return 1/2*np.trapz(np.trapz((self.RAR_MTF()*self.orbital_velocity_mtf()*self.PSI+np.conj(self.RAR_MTF()*self.orbital_velocity_mtf()*self.PSI))*np.exp(1j*self.wavenumbers*x),self.wavenumbers[0,:], axis=0),self.wavenumbers[:,0])

    def linear_mapping_transform(self):
        return 0.5*(abs(self.SAR_MTF())**2*self.PSI+(abs(np.rot90(self.SAR_MTF(),2))**2*np.rot90(self.PSI,2)))
    
    def quasilinear_mapping_transform(self):
        # i think i might be omiting the imag part
        # print(f"rho0 {self.v_covariance(0)}")
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        return np.exp(-self.kx**2*self.beta**2*self.f_v[0,0])*self.linear_mapping_transform()
    
    def nonlinear_mapping_transform(self, n):
        nonLinearTerms=np.zeros((self.N, self.N)).astype(np.complex64)
        for i in range(1,n+1):
            nonLinearTerms+=self.nonLinearity(i)
        import matplotlib.pyplot as plt
        # plt.figure()
        # plt.contour(abs(nonLinearTerms))
        # plt.show()
        # return nonLinearTerms
        return np.exp(-self.kx**2*self.beta**2*self.f_v[0,0])*nonLinearTerms
    
    def nonLinearity(self, n):
        import math
        if n==1:
            factorial_2=0
        else:
            factorial_2=1/math.factorial(n-2)
        return (2*np.pi)**(-2)*fftshift(fft2(1/math.factorial(n-1)*self.f_r*self.f_v**(n-1)+factorial_2*(self.f_rv-self.f_rv[0,0])*(np.rot90(self.f_rv,2)-self.f_rv[0,0])*self.f_r**(n-2)))
    
    def autocovariance(self, image, lag_x=0, lag_y=0):
        rows, cols = image.shape
        mean_image = np.mean(image)
        autocov = 0.0
        for i in range(rows - lag_y):
            for j in range(cols - lag_x):
                autocov += (image[i, j] - mean_image) * (image[i + lag_y, j + lag_x] - mean_image)
        autocov /= ((rows - lag_y) * (cols - lag_x))
        
        return autocov
    
    def crosscovariance(self, image1, image2, lag_x=0, lag_y=0):
        rows, cols = image1.shape
        mean_image1 = np.mean(image1)
        mean_image2 = np.mean(image2)
        cov = 0.0
        for i in range(rows - abs(lag_y)):
            for j in range(cols - abs(lag_x)):
                cov += (image1[i, j] - mean_image1) * (image2[i + lag_y, j + lag_x] - mean_image2)
        return cov / ((rows - abs(lag_y)) * (cols - abs(lag_x)))
        
