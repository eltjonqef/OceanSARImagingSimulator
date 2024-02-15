
import numpy as np
from multiprocessing import Pool
from scipy.fft import fft2, ifft2
from enum import Enum
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model
from scipy import linalg

class surfaceGenerator:
    def __init__(self, length, facet, wind_speed, wind_direction, seconds, timestep, fetch):
        self.g=9.81
        self.dx=facet
        self.L=length
        self.N=int(self.L/self.dx)
        self.x=np.linspace(-self.L/2,self.L/2,self.N)
        self.y=np.linspace(-self.L/2,self.L/2,self.N)
        self.time=np.linspace(0,seconds, int(seconds/timestep))
        self.surface=np.zeros((self.time.size, self.N, self.N))
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        
    
    def generateSurface(self):
        kx_s = (2*np.pi*np.fft.fftfreq(self.N, self.dx)).astype(np.float32)
        ky_s = (2*np.pi*np.fft.fftfreq(self.N, self.dx)).astype(np.float32)
        kx, ky = np.meshgrid(kx_s, ky_s)
        self.KX=kx
        self.KY=ky
        kx_res = kx[0, 1] - kx[0, 0]
        ky_res = ky[1, 0] - ky[0, 0]
        
        k = np.sqrt(kx**2 + ky**2)
        good_k = np.where(k > np.min(np.array([kx_res, ky_res])) / 2.0)
        kxn = np.zeros_like(kx, dtype=np.float32)
        kyn = np.zeros_like(kx, dtype=np.float32)
        kxn[good_k] = kx[good_k] / k[good_k]
        kyn[good_k] = ky[good_k] / k[good_k]
        kinv = np.zeros(k.shape, dtype=np.float32)
        kinv[good_k] = 1./k[good_k]
        theta = np.angle(np.exp(1j * (np.arctan2(ky, kx) -self.wind_direction))).astype(np.float32)
        self.omega = np.sqrt(np.float32(self.g) * k)
        S=omnidirectional_spectrum(spectrum_model.JONSWAP,k,self.wind_speed,F=self.fetch, good_k=good_k).getSpectrum()
        D=spreading_function(spreading_model.Longuet_Higgins,theta, 8, good_k=good_k).getSpread()
        wave_dirspec = (kinv) * S * D
        self.PSI=kinv*S*D
        np.random.seed(13)
        random_cg = (1./np.sqrt(2) * (np.random.normal(0., 1., size=[self.N, self.N]) +1j * np.random.normal(0., 1., size=[self.N, self.N]))).astype(np.complex64)
        self.wave_coefs=(self.N*self.N*np.sqrt(2.*wave_dirspec*kx_res*ky_res)*random_cg).astype(np.complex64)

    def generateTimeSeries(self):
        for frame, t in enumerate(self.time):
            wave_coefs_phased=(self.wave_coefs*np.exp(-1j*self.omega*t)).astype(np.complex64)
            self.surface[frame,:,:]=np.real(np.fft.ifft2(wave_coefs_phased))

    def getPSI(self):
        return self.PSI

    def getKx(self):
        return self.k
    def generate(self):
        self.generateSurface()
        self.generateTimeSeries()
        #self.frameNew()
        return self.surface