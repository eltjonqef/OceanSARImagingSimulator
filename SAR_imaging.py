import numpy as np
from enum import Enum
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model

class Polarization(Enum):
    Vertical=0
    Horizontal=1
    Both=2

class Band(Enum):
    X=0
    C=1
    L=2

class SAR_imaging:
    def __init__(self, surface, length, N, spectrum, incidence_angle, k, omega, wind_speed, wind_direction, fetch):
        self.polarization=Polarization.Horizontal
        self.H=693000
        self.V=7400
        self.R=self.H/np.cos(incidence_angle)
        self.beta=self.R/self.V
        self.L=length
        self.N=N
        self.dx=length/N
        self.surface=surface
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        self.incidence_angle=incidence_angle
        self.ky=k*np.sin((np.linspace(0, 2*np.pi,(N))))
        self.kx=k*np.cos((np.linspace(0, 2*np.pi,(N))))
        self.k=k
        self.theta=np.linspace(0, 2*np.pi, (N))
        self.spectrum=spectrum
        self.omega=omega
        self.mu=0.5
        self.light_speed=299792458
        self.frequency=5e9
        self.band=Band.C
        if self.band==Band.X:
            self.epsilon=49-35.5j
        elif self.band==Band.C:
            self.epsilon=60-36j
        elif self.band==Band.L:
            self.epsilon=72-59j

    def image(self):
        I=np.zeros((self.N,self.N))
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        sigma=self.NRCS()
        pa=1
        Ur=self.orbital_velocity()
        for i in range(self.N):
            for j in range(self.N):
                I[i, j]=np.trapz(sigma[i,:]/pa*np.exp(-np.pi**2*((x[i,j]-x[i,:]-self.beta*Ur[i,:])/pa)**2), dx=self.dx*self.dx)
        return I
    
    def average_NRCS(self, theta=None):
        if theta is None:
            theta=self.incidence_angle
        k_e=2*np.pi/(self.light_speed/self.frequency)
        k_brag=2*k_e*np.sin(theta)
        wavenumbers=np.logspace(-3,5,500)
        pos=np.abs(wavenumbers-k_brag).argmin()
        wavenumbers[pos]=k_brag
        PSI=omnidirectional_spectrum(spectrum=self.spectrum,k=wavenumbers,v=self.wind_speed,F=self.fetch).getSpectrum()[pos]
        sn=0#np.gradient(self.surface,axis=0)
        sp=0#np.gradient(self.surface,axis=1)
        theta_l=np.arccos(np.cos(theta-sp)*np.cos(sn))
        return 8*np.pi*k_e**4*np.cos(theta_l)**4*PSI*np.abs(self.complex_scattering())**2
    
    def NRCS(self):
        return self.average_NRCS()*(1+np.real(2*np.fft.ifft2(np.fft.ifftshift(self.RAR_MTF()*np.fft.fftshift(np.fft.fft2(self.surface))))))

    def complex_scattering(self):
        if self.polarization==Polarization.Vertical:
            return self.epsilon**2*(1+np.sin(self.incidence_angle)**2)/(self.epsilon*np.cos(self.incidence_angle)+np.sqrt(self.epsilon))**2
        elif self.polarization==Polarization.Horizontal:
            return self.epsilon/(np.cos(self.incidence_angle)+np.sqrt(self.epsilon))**2
        else:
            return None

    def tilt_mtf(self):
        return -(4/np.tan(self.incidence_angle))/(1+np.sin(self.incidence_angle)**2)*(1j*self.ky)
    
    def hydrodynamic_mtf(self):
        return 4.5*self.omega*(self.ky)**2*(self.omega-self.mu*1j)/(np.abs(self.k)*(self.omega**2+self.mu**2))
    
    def range_bunching_mtf(self):
        return -1j*self.ky/np.tan(self.incidence_angle)

    def velocity_bunching_mtf(self):
        return 1j*self.R/self.V*self.omega*(np.sin(self.incidence_angle)*np.cos(self.wind_direction)+1j*np.cos(self.incidence_angle))
    
    def orbital_velocity_mtf(self):
        return self.omega*(self.ky/self.k*np.sin(self.incidence_angle)+1j*np.cos(self.incidence_angle))

    def orbital_velocity(self):
        return 2*np.real(np.fft.ifft2(np.fft.ifftshift(self.orbital_velocity_mtf()*np.fft.fftshift(np.fft.fft2(self.surface)))))
    
    def RAR_MTF(self):
        return self.tilt_mtf()+self.hydrodynamic_mtf()+self.range_bunching_mtf()
    
