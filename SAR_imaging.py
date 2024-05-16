import numpy as np
from enum import Enum
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from scipy import special

class Polarization(Enum):
    Vertical=0
    Horizontal=1
    Both=2

class Band(Enum):
    X=0
    C=1
    L=2

class SAR_imaging:
    def __init__(self, surfaceGenerator, length, N, spectrum, incidence_angle, wind_speed, wind_direction, fetch, spatial_resolution, integration_time):
        self.random_phase=surfaceGenerator.random_phase
        self.PSI=surfaceGenerator.PSI
        self.wave_coeffs=surfaceGenerator.wave_coeffs
        self.polarization=Polarization.Vertical
        self.H=693000
        self.V=7400
        self.R=self.H/np.cos(incidence_angle)
        self.beta=100#self.R/self.V
        self.spatial_resolution=spatial_resolution
        print(self.beta)
        self.L=length
        self.N=N
        self.dx=length/N
        self.surface=surfaceGenerator.surface[0,:,:]
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        self.incidence_angle=incidence_angle
        self.integration_time=integration_time
        self.ky=surfaceGenerator.KY#*np.sin(np.pi/2)#*np.sin(np.linspace(0,2*np.pi, self.N))#wavenumbers*np.sin((np.linspace(0, 2*np.pi,(N))))
        self.kx=surfaceGenerator.KX#*np.cos(np.pi/2)#np.cos(np.linspace(0,2*np.pi, self.N))#wavenumbers*np.cos((np.linspace(0, 2*np.pi,(N))))
        self.wavenumbers=surfaceGenerator.K
        self.theta=np.linspace(0, 2*np.pi, (N))
        self.spectrum=spectrum
        self.dispersion_relation=surfaceGenerator.omega
        self.hydrodynamic_relaxation_rate=0.5
        self.light_speed=299792458
        self.frequency=5e9
        self.wavelength=self.light_speed/self.frequency
        self.k_e=2*np.pi/(self.light_speed/self.frequency)
        self.band=Band.C
        self.I=np.zeros((self.N,self.N))
        self.minus_PSI=surfaceGenerator.minus_PSI
        if self.band==Band.X:
            self.dielectric_constant=49-35.5j
        elif self.band==Band.C:
            self.dielectric_constant=60-36j
        elif self.band==Band.L:
            self.dielectric_constant=72-59j

    def generate(self):
        self.orbital_velocity()
        # self.orbital_velocity_sum()
        self.image()

    def image(self):
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        sigma=self.NRCS()
        Ur=self.u_r
        print(f"Variance of Obrital Velocities {np.var(Ur)}")
        # print(f"rho0 {self.v_covariance(0)[0,0]} max {np.max(self.v_covariance(0))}")
        # print(f"azimiuhth resolution {self.azimuth_resolution()}")
        pa=self.degraded_azimuthal_resolution()
        print(f"degraged resolution {pa}")
        # print(f"rho a {pa}")
        for i in range(self.N):
            for j in range(self.N):
                self.I[i, j]=np.pi*self.integration_time**2*self.azimuth_resolution()/2*np.trapz(sigma[i,:]/pa*np.exp(-(np.pi/pa)**2*((x[i,j]-x[i,:]-self.beta*Ur[i,:]))**2), dx=self.dx*self.dx)
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
        print(f"psi {np.cos(theta_l)}")
        return 8*np.pi*self.k_e**4*np.cos(theta_l)**4*PSI*np.abs(self.complex_scattering())**2
    
    def NRCS(self):
        print(f"average nrcs {self.average_NRCS()}")
        return self.average_NRCS()*(1+np.real(2*np.fft.ifft2(np.fft.ifftshift(self.RAR_MTF()*np.fft.fftshift(np.fft.fft2(self.surface))+np.conj(self.RAR_MTF()*np.fft.fftshift(np.fft.fft2(self.surface)))))))

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
        return 1j*self.R/self.V*self.kx*self.dispersion_relation*(np.sin(self.incidence_angle)*np.cos(self.wind_direction)+1j*np.cos(self.incidence_angle))
    
    def orbital_velocity_mtf(self, negative=False):
        if negative:
            return -self.dispersion_relation*(self.kx/-self.wavenumbers*np.sin(self.incidence_angle)*1j+np.cos(self.incidence_angle))
        return -self.dispersion_relation*(self.kx/self.wavenumbers*np.sin(self.incidence_angle)*1j+np.cos(self.incidence_angle))
    
    def orbital_velocity(self):
        self.u_r=np.real(np.fft.ifft2(np.fft.ifftshift(self.orbital_velocity_mtf()*np.fft.fftshift(np.fft.fft2(self.surface)))))
        return

    def orbital_velocity_sum(self):
        x, y=np.meshgrid(np.linspace(-self.L/2,self.L/2,self.N), np.linspace(-self.L/2,self.L/2,self.N))
        g=9.81
        u_x=np.zeros((self.N,self.N))
        u_y=np.zeros((self.N,self.N))
        u_z=np.zeros((self.N,self.N))
        self.hta=np.zeros((self.N,self.N))

        print(self.wave_coeffs.shape)
        # Check shape and dtype of self.wave_coeffs
        wave_coeffs=np.abs(self.wave_coeffs)/self.N**2
        print("Shape of self.wave_coeffs:", wave_coeffs.shape)
        print("Dtype of self.wave_coeffs:", wave_coeffs.dtype)
        # aaa=wave_coeffs[]
        #np.cos(k*(self.x*np.cos(self.wind_direction)+self.y*np.sin(self.wind_direction))-self.omega[i]*t)
        for i in range(self.N):
            for j in range(self.N):
                self.hta+=wave_coeffs[i,j]*np.cos(x*self.kx[i,j]+y*self.ky[i,j]+self.random_phase[i,j])
                u_x+=wave_coeffs[i,j]*self.dispersion_relation[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.cos(self.wind_direction)
                u_y+=wave_coeffs[i,j]*self.dispersion_relation[i,j]*np.cos(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])*np.sin(self.wind_direction)
                u_z+=wave_coeffs[i,j]*self.dispersion_relation[i,j]*np.sin(self.kx[i,j]*x+self.ky[i,j]*y+self.random_phase[i,j])
            print(i)
        # u_x*=g
        # u_y*=g
        # u_z*=g

        self.u_r_sum=u_z*np.cos(self.incidence_angle)-np.sin(self.incidence_angle)*(u_x*np.sin(self.wind_direction)+u_y*np.cos(self.wind_direction))

    def mean_orbital_velocity(self):
        B_f=2/(self.kx*self.dx)*np.sin(self.kx*self.dx/2)*2/(self.ky*self.dx)*np.sin(self.ky*self.dx/2)*2/(self.dispersion_relation*self.integration_time)*np.sin(self.dispersion_relation*self.integration_time/2)
        return np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(self.u_r))*B_f)))


    
    def orbital_acceleration_mtf(self):
        return self.dispersion_relation**2*(self.ky/self.wavenumbers*np.sin(self.incidence_angle)+1j*np.cos(self.incidence_angle))

    def RAR_MTF(self):
        return self.tilt_mtf()+self.hydrodynamic_mtf()+self.range_bunching_mtf()
    
    def SAR_MTF(self):
        return self.RAR_MTF()+self.velocity_bunching_mtf()
    

    def coherence_time(self):
        wind_speed_19_5=self.wind_speed*(19.5/10)**(1/7)
        print(f"wind speed {wind_speed_19_5}")
        return 3*self.wavelength/wind_speed_19_5*special.erf(2.7*self.spatial_resolution/wind_speed_19_5**2)**(-1/2)
    
    def azimuth_resolution(self):
        lamda_e=2*np.pi/self.k_e
        # print(f"azimuth resolution {(lamda_e*self.R)/(2*self.V*self.integration_time)}")
        return 2
        return (lamda_e*self.R)/(2*self.V*self.integration_time)

    def degraded_azimuthal_resolution(self):
        # print(np.sqrt(1+self.integration_time**2/self.coherence_time()**2))
        # print(self.integration_time)
        # print(self.coherence_time())
        print(f"integration time {self.integration_time}")
        print(f"coherence time {self.coherence_time()}")
        # return 2
        return self.azimuth_resolution()*np.sqrt(1+(self.integration_time/self.coherence_time())**2)
    
    def wave_field(self):
        return 2*np.abs(self.SAR_MTF())**2*np.fft.fftshift(np.fft.fft2(self.I))
        return np.abs(np.fft.fftshift(np.fft.fft2(self.surface))) #1/2*np.real(2*np.abs(self.SAR_MTF())**2*np.fft.fftshift(np.fft.fft2(self.I)))

    def noisy_image(self):
        noise=np.random.exponential(scale=1,size=(self.N, self.N))
        return self.I*noise
    
    def v_covariance(self,x=0):
        return np.trapz(np.trapz((abs(self.orbital_velocity_mtf()))**2*self.PSI,self.kx[0,:],axis=0),self.ky[:,0],axis=0)*np.exp(1j*self.wavenumbers*np.sqrt(x))
        # x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        #USING NOWreturn np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((abs(self.orbital_velocity_mtf())**2)*(self.PSI)))))
        # return 0.5*np.real(np.trapz(np.trapz((abs(self.orbital_velocity_mtf())**2)*(self.PSI),self.kx[0,:], axis=0), self.ky[:,0], axis=0))*np.exp(1j*(self.wavenumbers*x+self.wavenumbers*y))
        # rho = real(fft.ifft2((abs(fft.fftshift(T_ku))**2*fft.fftshift(PSI)),(Nx,Ny)))
        # return np.real(np.fft.ifft2((abs(np.fft.fftshift(self.orbital_velocity_mtf()))**2*(self.PSI))))#*self.L**2/(2*np.pi)**2
        # return np.real(np.trapz(np.trapz((abs(self.orbital_velocity_mtf())**2*self.PSI+np.conj(abs(self.orbital_velocity_mtf())**2*self.PSI))*np.exp(1j*self.wavenumbers*x),self.wavenumbers[0,:], axis=0),self.wavenumbers[:,0]))

    def R_covariance(self,x):
        return 1/2*np.trapz(np.trapz((abs(self.RAR_MTF())**2*self.PSI+np.conj(abs(self.RAR_MTF())**2*self.PSI))*np.exp(1j*self.wavenumbers*x),self.wavenumbers[0,:], axis=0),self.wavenumbers[:,0])

    def Rv_covariance(self,x):
        return 1/2*np.trapz(np.trapz((self.RAR_MTF()*self.orbital_velocity_mtf()*self.PSI+np.conj(self.RAR_MTF()*self.orbital_velocity_mtf()*self.PSI))*np.exp(1j*self.wavenumbers*x),self.wavenumbers[0,:], axis=0),self.wavenumbers[:,0])

    def inverse_linear_transform(self):
        return 0.5*(abs(self.SAR_MTF())**2*self.PSI+(abs(self.SAR_MTF())**2*np.rot90(self.PSI,2)))

    def inverse_quasilinear_transform(self):
        # i think i might be omiting the imag part
        # print(f"rho0 {self.v_covariance(0)}")
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        print(f"ql coeff {self.beta**2}")
        return np.exp(-self.kx**2*self.beta**2*np.var(self.u_r))*self.inverse_linear_transform()

    def inverse_nonlinear_transform(self):
        x, y=np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        ksi=self.beta**2*self.v_covariance(0)
        return 1/(4*np.pi**2)*np.exp(-self.kx**2*ksi)*np.trapz(np.exp(self.kx**2*self.beta**2*self.v_covariance(x))*(1+self.R_covariance(x)+1j*self.beta*(self.Rv_covariance(x)-self.Rv_covariance(-x))+(self.kx*self.beta)**2*(self.Rv_covariance(x)-self.Rv_covariance(0))*(self.Rv_covariance(-x)-self.Rv_covariance(0))),dx=self.dx*self.dx)