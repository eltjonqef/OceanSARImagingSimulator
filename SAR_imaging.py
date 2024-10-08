import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model
from scipy import special
from parameters import parameters
from surface import surfaceGenerator
from frequencyEnums import Band, Polarization
import math

class SAR_imaging:
    def __init__(self, surfaceGenerator:surfaceGenerator, params:parameters):
        self.random_phase=surfaceGenerator.random_phase
        self.PSI=surfaceGenerator.PSI
        self.wave_coeffs=surfaceGenerator.wave_coeffs
        self.polarization=params.polarization
        self.H=params.H
        self.V=params.V
        self.R=self.H/np.cos(params.incidence_angle)
        self.beta=params.beta
        if params.beta==0:
            self.beta=self.R/self.V
        self.dx=params.length_x/params.N_x
        self.dy=params.length_y/params.N_y
        self.L_x=params.length_x
        self.L_y=params.length_y
        self.N_x=params.N_x
        self.N_y=params.N_y
        self.surface=surfaceGenerator.surface[0,:,:]
        self.Z=fftshift(fft2(self.surface))
        self.wind_speed=params.wind_speed
        self.wind_direction=params.wind_direction
        self.fetch=params.fetch
        self.incidence_angle=params.incidence_angle
        self.azimuth_resolution=params.azimuth_resolution
        self.range_resolution=params.range_resolution
        self.spatial_resolution=self.azimuth_resolution
        self.ky=surfaceGenerator.KY
        self.kx=surfaceGenerator.KX
        self.wavenumbers=surfaceGenerator.K
        self.spectrum=params.spectrum
        self.spreading=params.spreading
        self.dispersion_relation=surfaceGenerator.omega
        self.hydrodynamic_relaxation_rate=0.5
        self.light_speed=299792458
        self.frequency=params.frequency
        self.wavelength=self.light_speed/self.frequency
        self.integration_time=(self.wavelength*self.beta)/(2*self.azimuth_resolution)#integration_time
        self.k_e=2*np.pi/(self.light_speed/self.frequency)
        if self.frequency>=1e9 and self.frequency<2e9:
            self.band=Band.L
        elif self.frequency>=4e9 and self.frequency<8e9:
            self.band=Band.C
        elif self.frequency>=8e9 and self.frequency<12e9:
            self.band=Band.X
        self.I=np.zeros((self.N_y,self.N_x))
        self.n=params.n
        self.S=params.S
        self.elfouhaily_k=params.elfouhaily_k
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
        self.add_noise()
        self.v_covariance()
        self.Rv_covariance()
        self.R_covariance()

    def image(self):
        x, y=np.meshgrid(np.linspace(0, self.L_x, self.N_x), np.linspace(0, self.L_y, self.N_y))
        Ur=self.u_r
        pa=self.degraded_azimuthal_resolution()
        for i in range(self.N_y):
            for j in range(self.N_x):
                self.I[i, j]=np.pi*self.integration_time**2*self.azimuth_resolution/2*np.trapz(self.sigma[i,:]/pa*np.exp(-(np.pi/pa)**2*((x[i,j]-x[i,:]-self.beta*Ur[i,:]))**2), dx=self.dx*self.dy)
        self.I=self.I-np.mean(self.I)

    
    def average_NRCS(self, theta=None):
        if theta is None:
            theta=self.incidence_angle
        kx_brag=2*self.k_e*np.sin(theta)
        ky_brag=2*self.k_e*np.cos(theta)
        k_brag=np.sqrt(kx_brag**2+ky_brag**2)
        theta_brag=np.angle(np.exp(1j * (np.arctan2(ky_brag, kx_brag)))).astype(np.float32)
        S=omnidirectional_spectrum(spectrum=self.spectrum,k=k_brag,v=self.wind_speed,F=self.fetch).getSpectrum()
        D=spreading_function(function=self.spreading, theta=theta_brag, wind_direction=self.wind_direction,n=self.n, S=self.S, F=self.fetch, k=self.elfouhaily_k, v=self.wind_speed, good_k=None).getSpread()
        PSI=S*D
        sn=0#np.gradient(self.surface,axis=0)
        sp=0#np.gradient(self.surface,axis=1)
        theta_l=np.arccos(np.cos(theta))
        # return 1
        return 8*np.pi*self.k_e**4*np.cos(theta_l)**4*PSI*np.abs(self.complex_scattering(theta_l))**2
    
    def NRCS(self):
        self.sigma=self.average_NRCS()*(1+np.real(2*ifft2(ifftshift(self.RAR_MTF()*self.Z+np.conj(self.RAR_MTF()*self.Z)))))

    def complex_scattering(self, theta_l):
        sn=np.pi/2#np.gradient(self.surface,axis=0)
        sp=0#np.gradient(self.surface,axis=1)
        self.polarization=Polarization.Vertical
        if self.polarization==Polarization.Vertical:
            return (np.sin(self.incidence_angle+sp)*np.cos(sp)/np.sin(theta_l))**2*(self.dielectric_constant**2*(1+np.sin(theta_l)**2)/(self.dielectric_constant*np.cos(theta_l)+np.sqrt(self.dielectric_constant))**2)
        elif self.polarization==Polarization.Horizontal:
            return (np.sin(sn)/np.sin(theta_l))**2*(self.dielectric_constant/(np.cos(theta_l)+np.sqrt(self.dielectric_constant))**2)
        else:
            return None

    def tilt_mtf(self):
        return -(4/np.tan(self.incidence_angle))/(1+np.sin(self.incidence_angle)**2)*(1j*self.ky)
    
    def hydrodynamic_mtf(self):
        return 4.5*self.dispersion_relation*(self.ky)**2*(self.dispersion_relation-self.hydrodynamic_relaxation_rate*1j)/(np.abs(self.wavenumbers)*(self.dispersion_relation**2+self.hydrodynamic_relaxation_rate**2))
    
    def range_bunching_mtf(self):
        return -1j*self.ky/np.tan(self.incidence_angle)

    def velocity_bunching_mtf(self):
        return -1j*self.beta*self.kx*self.orbital_velocity_mtf()
    
    def orbital_velocity_mtf(self):
        return -self.dispersion_relation*(self.ky/self.wavenumbers*np.sin(self.incidence_angle)*1j+np.cos(self.incidence_angle))
    
    def orbital_velocity(self):
        self.u_r=np.real(ifft2(ifftshift(self.orbital_velocity_mtf()*self.Z)))

    def orbital_velocity_sum(self):
        x, y=np.meshgrid(np.linspace(-self.L_x/2,self.L_x/2,self.N_x), np.linspace(-self.L_y/2,self.L_y/2,self.N_y))
        g=9.81
        u_x=np.zeros((self.N_x,self.N_y))
        u_y=np.zeros((self.N_x,self.N_y))
        u_z=np.zeros((self.N_x,self.N_y))
        self.hta=np.zeros((self.N_x,self.N_y))
        for i in range(self.N_x):
            for j in range(self.N_y):
                self.hta+=self.wave_coeffs[i,j]*np.cos(x*self.kx[i,j]+y*self.ky[i,j]+self.random_phase[i,j])
                u_x+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.cos(self.wind_direction)
                u_y+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.sin(self.wind_direction)
                u_z+=self.wave_coeffs[i,j]/self.dispersion_relation[i,j]*self.wavenumbers[i,j]*np.sin(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])
            print(i)
        u_x*=g
        u_y*=g
        u_z*=g
        self.u_r_sum=u_z*np.cos(self.incidence_angle)-np.sin(self.incidence_angle)*(u_x*np.sin(self.wind_direction)+u_y*np.cos(self.wind_direction))
    
    def orbital_acceleration_mtf(self):
        return self.dispersion_relation**2*(self.ky/self.wavenumbers*np.sin(self.incidence_angle)+1j*np.cos(self.incidence_angle))

    def RAR_MTF(self):
        return self.tilt_mtf()+self.hydrodynamic_mtf()+self.range_bunching_mtf()
    
    def SAR_MTF(self):
        return self.RAR_MTF()+self.velocity_bunching_mtf()
    
    def coherence_time(self):
        wind_speed_19_5=self.wind_speed*(19.5/10)**(1/7)
        return 3*self.wavelength/wind_speed_19_5*special.erf(2.7*self.spatial_resolution/wind_speed_19_5**2)**(-1/2)

    def degraded_azimuthal_resolution(self):
        return self.azimuth_resolution*np.sqrt(1+(self.integration_time/self.coherence_time())**2)
    
    def add_noise(self):
        noise=np.random.exponential(scale=1,size=(self.N_y, self.N_x))
        self.noisy_I=self.I*noise
    
    def v_covariance(self):
        Fk=self.PSI*(2*np.pi/self.dx)*(2*np.pi/self.dy)
        self.f_v=np.real(ifft2(ifftshift(abs(self.orbital_velocity_mtf())**2*Fk)))

    def R_covariance(self):
        Fk=self.PSI*(2*np.pi/self.dx)*(2*np.pi/self.dy)
        self.f_r=0.5*(ifft2(ifftshift(abs(self.RAR_MTF())**2*Fk+abs(np.rot90(self.RAR_MTF(),2))**2*np.rot90(Fk,2))))

    def Rv_covariance(self):
        Fk=self.PSI*(2*np.pi/self.dx)*(2*np.pi/self.dy)
        self.f_rv=0.5*np.real(ifft2(ifftshift(self.RAR_MTF()*np.conj(self.orbital_velocity_mtf())*Fk+np.conj(np.rot90(self.RAR_MTF(),2))*np.rot90(self.orbital_velocity_mtf(),2)*np.rot90(Fk,2))))

    def linear_mapping_transform(self):
        Fk=self.PSI*(2*np.pi/self.dx)*(2*np.pi/self.dy)
        return 0.5*(abs(self.SAR_MTF())**2*Fk+abs(np.rot90(self.SAR_MTF(),2))**2*np.rot90(Fk,2))
    
    def quasilinear_mapping_transform(self):
        return np.exp(-self.kx**2*self.beta**2*self.f_v[0,0])*self.linear_mapping_transform()
    
    def nonlinear_mapping_transform(self, n):
        nonLinearTerms=np.zeros((self.N_y, self.N_x)).astype(np.complex64)
        for i in range(1,n+1):
            nonLinearTerms+=self.nonLinearity(i)
        return np.exp(-self.kx**2*self.beta**2*self.f_v[0,0])*nonLinearTerms
    
    def nonLinearity(self,i):
        nonLinearTerms=np.zeros((self.N_y, self.N_x)).astype(np.complex64)
        nonLinearTerms+=(self.kx*self.beta)**(2*i-2)*self.P_2n2(i)
        nonLinearTerms+=(self.kx*self.beta)**(2*i-1)*self.P_2n1(i)
        nonLinearTerms+=(self.kx*self.beta)**(2*i)*self.P_2n(i)
        return nonLinearTerms
    
    def P_2n(self,n):
        return (2*np.pi)**(-2)*fftshift(fft2((self.f_v**n)/math.factorial(n)))
    
    def P_2n1(self,n):
        return (2*np.pi)**(-2)*fftshift(fft2((1j*(self.f_rv-np.rot90(self.f_rv,2))*self.f_v**(n-1))/math.factorial(n-1)))

    def P_2n2(self, n):
        if n==1:
            factorial_2=0
        else:
            factorial_2=1/math.factorial(n-2)
        return (2*np.pi)**(-2)*fftshift(fft2(1/math.factorial(n-1)*self.f_r*self.f_v**(n-1)+factorial_2*(self.f_rv-self.f_rv[0,0])*(np.rot90(self.f_rv,2)-self.f_rv[0,0])*self.f_r**(n-2)))