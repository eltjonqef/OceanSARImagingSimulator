
import numpy as np
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model

class surfaceGenerator:
    def __init__(self, spectrum, spreading, length, N, wind_speed, wind_direction, n, S, seconds, timestep, fetch, elfouhaily_k):
        self.spectrum=spectrum
        self.omnidirectional_spectrum=None
        self.spreading_function=None
        self.spreading=spreading
        self.g=9.81
        self.dx=length/N
        self.L=length
        self.N=N
        self.n=n
        self.S=S
        self.elfouhaily_k=elfouhaily_k
        print(self.N)
        print(self.dx)
        print(self.L)
        self.x=np.linspace(-self.L/2,self.L/2,self.N)
        self.y=np.linspace(-self.L/2,self.L/2,self.N)
        self.time=np.linspace(0,seconds, int(seconds/timestep))
        self.surface=np.zeros((self.time.size, self.N, self.N))
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        
    
    def generateSurface(self):
        kx_s = np.fft.fftshift((2*np.pi*np.fft.fftfreq(self.N, self.dx)).astype(np.float32))
        ky_s = np.fft.fftshift((2*np.pi*np.fft.fftfreq(self.N, self.dx)).astype(np.float32))
        kx, ky = np.meshgrid(kx_s, ky_s)
        kx_res = kx[0, 1] - kx[0, 0]
        ky_res = ky[1, 0] - ky[0, 0]
        # if self.monochromatic:
        MONOCHROMATIC=False
        if MONOCHROMATIC:
            # self.K[150,129]=k_tmp[128,160]
            x=150
            y=255
            tmp=kx[x,y]
            kx=np.zeros((self.N,self.N))
            ky=np.zeros((self.N,self.N))
            kx[x,y]=tmp
            ky[x,y]=tmp

        self.KX=kx
        self.KY=ky
        
        k_tmp = np.sqrt(kx**2 + ky**2)
        good_k = np.where(k_tmp > np.min(np.array([kx_res, ky_res])) / 2.0)
        self.K=np.zeros(k_tmp.shape, dtype=np.float32)
        self.K[good_k]=k_tmp[good_k]
        # self.K=k_tmp
        # self.K[self.K==0]=0.000000000001#np.finfo(float).tiny
        kxn = np.zeros_like(kx, dtype=np.float32)
        kyn = np.zeros_like(kx, dtype=np.float32)
        kxn[good_k] = kx[good_k] #/ self.K[good_k]
        kyn[good_k] = ky[good_k]#/ self.K[good_k]
        # kx=kxn
        # ky=kyn
        kinv = np.zeros(self.K.shape, dtype=np.float32)
        kinv[good_k] = 1./self.K[good_k]
        self.theta = np.angle(np.exp(1j * (np.arctan2(ky, kx) -self.wind_direction))).astype(np.float32)
        # import matplotlib.pyplot as plt
        # plt.plot(self.theta)
        # plt.show()
        self.omnidirectional_spectrum=omnidirectional_spectrum(spectrum=self.spectrum,k=self.K,v=self.wind_speed,F=self.fetch, good_k=None)
        # self.omnidirectional_spectrum.plot()
        self.spreading_function=spreading_function(function=self.spreading, theta=self.theta, n=self.n, S=self.S, F=self.fetch, k=self.elfouhaily_k, v=self.wind_speed, good_k=None)
        # self.spreading_function.plot()
        S=self.omnidirectional_spectrum.getSpectrum()
        # S[150,150]=500
        S[np.isnan(S)]=0
        self.KX[self.KX==0]=0.00000001
        self.KY[self.KY==0]=0.00000001
        self.K[self.K==0]=0.00000001
        self.omega = np.sqrt(np.float32(self.g) * self.K)
        D=self.spreading_function.getSpread()
        wave_dirspec = (kinv) * S * D
        self.PSI=kinv*S*D

        self.random_cg = (1./np.sqrt(2) * (np.random.normal(0., 1., size=[self.N, self.N]) +1j * np.random.normal(0., 1., size=[self.N, self.N]))).astype(np.complex64)
        self.wave_coeffs=(self.N*self.N*np.sqrt(2.*wave_dirspec*kx_res*ky_res)*self.random_cg).astype(np.complex64)

    def generateTimeSeries(self):
        for frame, t in enumerate(self.time):
            wave_coefs_phased=(self.wave_coeffs*np.exp(-1j*self.omega*t)).astype(np.complex64)
            self.surface[frame,:,:]=np.real(np.fft.ifft2(np.fft.ifftshift(wave_coefs_phased)))
        

    
    def generate(self):
        self.generateSurface()
        self.generateTimeSeries()
        #self.frameNew()
        return self.surface


    def getSurfaceVariances(self):
        return [np.var(self.surface[frame,:,:]) for frame, _ in enumerate(self.time)]
    
    def getSpectrumIntegral(self):
        return np.trapz(np.trapz(self.PSI,self.KX[0,:], axis=0), self.KY[:,0], axis=0)
    
    def getSlopes(self):
        Sx=np.gradient(self.surface[0,:,:],axis=0).flatten()#[np.gradient(self.surface[frame,:,:],axis=0).flatten() for frame, _ in enumerate(self.time)]
        Sy=np.gradient(self.surface[0,:,:],axis=1).flatten()#[np.gradient(self.surface[frame,:,:],axis=1).flatten() for frame, _ in enumerate(self.time)]
        return Sx, Sy

    def getSlopesVariance(self):
        Sx=[np.gradient(self.surface[frame,:,:],axis=1).flatten() for frame, _ in enumerate(self.time)]
        Sy=[np.gradient(self.surface[frame,:,:],axis=0).flatten() for frame, _ in enumerate(self.time)]
        return [np.var(Sx[frame])+np.var(Sy[frame]) for frame, _ in enumerate(self.time)]

    def getSlopeIntegral(self):
        return np.trapz(np.trapz(np.power(self.KX[0,:],2).reshape(-1,1)*self.PSI,self.KX[0,:], axis=0), self.KY[:,0], axis=0)
    
    def getSignificantWaveHeights(self):
        return [4*np.sqrt(np.var(self.surface[frame,:,:])) for frame, _ in enumerate(self.time)]