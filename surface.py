
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift, fftfreq
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model
from parameters import parameters
class surfaceGenerator:
    def __init__(self, params:parameters):
        self.spectrum=params.spectrum
        self.omnidirectional_spectrum=None
        self.spreading_function=None
        self.spreading=params.spreading
        self.g=9.81
        self.dx=params.length_x/params.N_x
        self.dy=params.length_y/params.N_y
        self.L_x=params.length_x
        self.L_y=params.length_y
        self.N_x=params.N_x
        self.N_y=params.N_y
        self.n=params.n
        self.S=params.S
        self.elfouhaily_k=params.elfouhaily_k

        self.x=np.linspace(-self.L_x/2,self.L_x/2,self.N_x)
        self.y=np.linspace(-self.L_y/2,self.L_y/2,self.N_y)
        self.time=np.linspace(0,params.seconds, int(params.seconds/params.timestep))
        self.surface=np.zeros((self.time.size, self.N_y, self.N_x))
        self.wind_speed=params.wind_speed
        self.wind_direction=params.wind_direction
        self.fetch=params.fetch
        self.MONOCHROMATIC=params.MONOCHROMATIC

    
    def generateSurface(self):
        kx_s = fftshift((2*np.pi*fftfreq(self.N_x, self.dx)).astype(np.float32))
        ky_s = fftshift((2*np.pi*fftfreq(self.N_y, self.dy)).astype(np.float32))
        kx, ky = np.meshgrid(kx_s, ky_s)
        kx_res = kx[0, 1] - kx[0, 0]
        ky_res = ky[1, 0] - ky[0, 0]
        self.Ks=kx_res
        print(f"kx_s {kx_s[1]-kx_s[0]}, kx_res {kx_res}")
        if self.MONOCHROMATIC:
            x=self.N//2+10
            y=self.N//2
            tmp=kx[x,y]
            kx=np.zeros((self.N_x,self.N_y))
            ky=np.zeros((self.N_x,self.N_y))
            kx[x,y]=0.01
            ky[x,y]=0
        self.KX=kx
        self.KY=ky
        k_tmp = np.sqrt(kx**2 + ky**2)
        good_k = np.where(k_tmp > np.min(np.array([kx_res, ky_res])) / 2.0)
        self.K=np.zeros(k_tmp.shape, dtype=np.float32)
        self.K[good_k]=k_tmp[good_k]
        kinv = np.zeros(self.K.shape, dtype=np.float32)
        kinv[good_k] = 1./self.K[good_k]
        self.theta = np.angle(np.exp(1j * (np.arctan2(ky, kx)))).astype(np.float32)
        self.omnidirectional_spectrum=omnidirectional_spectrum(spectrum=self.spectrum,k=self.K,v=self.wind_speed,F=self.fetch, good_k=None)
        self.spreading_function=spreading_function(function=self.spreading, theta=self.theta, wind_direction=self.wind_direction,n=self.n, S=self.S, F=self.fetch, k=self.elfouhaily_k, v=self.wind_speed, good_k=None)
        S=self.omnidirectional_spectrum.getSpectrum()
        S[np.isnan(S)]=0
        self.KX[self.KX==0]=0.00000001
        self.KY[self.KY==0]=0.00000001
        self.K[self.K==0]=0.00000001
        self.omega = np.sqrt(np.float32(self.g) * self.K)
        D=self.spreading_function.getSpread()
        wave_dirspec = kinv*S * D
        self.PSI=kinv*S*D
        np.random.seed(10)
        self.random_cg = (1./np.sqrt(2) * (np.random.normal(0., 1., size=[self.N_y, self.N_x]) +1j * np.random.normal(0., 1., size=[self.N_y, self.N_x]))).astype(np.complex64)
        self.random_phase=np.angle(self.random_cg)
        self.wave_coeffs=(np.sqrt(2.*wave_dirspec*kx_res*ky_res))

    def generateTimeSeries(self):
        for frame, t in enumerate(self.time):
            wave_coefs_phased=(self.N_x*self.N_y*self.wave_coeffs*self.random_cg*np.exp(-1j*self.omega*t)).astype(np.complex64)
            self.surface[frame,:,:]=np.real(ifft2(ifftshift(wave_coefs_phased)))
        

    
    def generate(self):
        self.generateSurface()
        self.generateTimeSeries()
        #self.frameNew()
        return self.surface


    def getSurfaceVariances(self):
        return [np.var(self.surface[frame,:,:]) for frame, _ in enumerate(self.time)]
    
    def getSpectrumIntegral(self):
        return np.trapz(np.trapz(self.PSI,self.KX[0,:], axis=1), self.KY[:,0], axis=0)
    
    def getSlopes(self):
        Sx=np.gradient(self.surface[0,:,:],axis=0).flatten()#[np.gradient(self.surface[frame,:,:],axis=0).flatten() for frame, _ in enumerate(self.time)]
        Sy=np.gradient(self.surface[0,:,:],axis=1).flatten()#[np.gradient(self.surface[frame,:,:],axis=1).flatten() for frame, _ in enumerate(self.time)]
        return Sx, Sy

    def getSlopesVariance(self):
        Sx=[np.gradient(self.surface[frame,:,:],axis=1).flatten() for frame, _ in enumerate(self.time)]
        Sy=[np.gradient(self.surface[frame,:,:],axis=0).flatten() for frame, _ in enumerate(self.time)]
        return [np.var(Sx[frame])+np.var(Sy[frame]) for frame, _ in enumerate(self.time)]

    def getSlopeIntegral(self):
        return np.trapz(np.trapz(self.K**2*self.PSI,self.KX[0,:], axis=1), self.KY[:,0], axis=0)
    
    def getSignificantWaveHeights(self):
        return [4*np.sqrt(np.var(self.surface[frame,:,:])) for frame, _ in enumerate(self.time)]