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
    def __init__(self, surface, spectrum, incidence_angle, k, omega, wind_speed, wind_direction, fetch):
        self.polarization=Polarization.Horizontal
        self.R=4e5
        self.V=8e6
        self.surface=surface
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        self.incidence_angle=incidence_angle
        self.ky=k*np.sin((np.linspace(0, 2*np.pi,(k.shape[0]))))
        self.kx=k*np.cos((np.linspace(0, 2*np.pi,(k.shape[0]))))
        self.k=k
        self.theta=np.linspace(0, 2*np.pi, len(k))
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

    # def average_NRCS(self):
    #     k_e=2*np.pi/(self.light_speed/self.frequency)
    #     k_brag=2*k_e*np.sin(self.incidence_angle)
    #     wavenumbers=np.logspace(-3,5,500)
    #     PSI=omnidirectional_spectrum(spectrum=self.spectrum,k=k_brag,v=self.wind_speed,F=self.fetch).getSpectrum()#omnidirectional_spectrum(spectrum=self.spectrum,k=wavenumbers,v=self.wind_speed,F=self.fetch).getSpectrum()[np.abs(wavenumbers-k_brag).argmin()]
    #     sn=np.gradient(self.surface,axis=0)
    #     sp=np.gradient(self.surface,axis=1)
    #     theta_l=np.arccos(np.cos(self.incidence_angle-sp)*np.cos(sn))
    #     return 8*np.pi*k_e**2*np.cos(theta_l)**4*PSI*self.complex_scattering()
    
    def average_NRCS(self, theta=None):
        if theta==None:
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
        return (self.average_NRCS()*(1+np.fft.ifft2(self.modulation_transfer_function()*np.fft.fftshift(np.fft.fft2(self.surface))).real))

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
        return 4.5*self.omega*(self.ky)**2*(self.omega-self.mu*1j)/(self.k*(self.omega**2+self.mu**2))
    
    def range_bunching_mtf(self):
        return -1j*self.ky/np.tan(self.incidence_angle)

    def velocity_bunching_mtf(self):
        return 1j*self.R/self.V*self.omega*(np.sin(self.incidence_angle)*np.cos(self.wind_direction)+1j*np.cos(self.incidence_angle))
    
    def orbital_velocity_mtf(self):
        return self.omega*(self.ky/self.k*np.sin(self.incidence_angle)+1j*np.cos(self.incidence_angle))

    def orbital_velocity(self):
        return 2*np.real(np.fft.ifft2(np.fft.ifftshift(self.orbital_velocity_mtf()*np.fft.fftshift(np.fft.fft2(self.surface)))))
    
    def orbintal_velocity_derivative(self):
        from superImposition_surface import superImpositionSurfaceGenerator

        superImpositionSurfaceGenerator=superImpositionSurfaceGenerator(spectrum, spreading, length, N, wind_speed, wind_direction, n, S, 1, 1, fetch, elfouhaily_k)
    
    def modulation_transfer_function(self):
        return self.tilt_mtf()+self.hydrodynamic_mtf()+self.range_bunching_mtf()+self.velocity_bunching_mtf()
    
