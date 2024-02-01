
import numpy as np
from multiprocessing import Pool
from scipy.fft import fft2, ifft2
class surfaceGenerator:
    def __init__(self, spectrum, spreading, wavenumbers, theta, size, facet, seconds):
        g=9.80665
        self.facet=facet
        self.spectrum=spectrum
        self.spreading=spreading
        self.delta_k=np.diff(wavenumbers, prepend=wavenumbers[0])
        self.delta_theta=np.diff(theta, prepend=theta[0])
        self.wavenumbers=wavenumbers
        self.theta=theta
        self.sinTheta=np.sin(self.theta)
        self.cosTheta=np.cos(self.theta)
        self.epsilon=np.random.uniform(0,2*np.pi,(len(wavenumbers),len(theta)))
        self.xLength=int(size[0]/facet)#np.linspace(0, size[0], int(size[0]/facet))
        self.yLength=int(size[1]/facet)#np.tile(np.linspace(0, size[1], int(size[1]/facet)),int(size[0]/facet)).reshape((-1,1))
        x=np.linspace(0, size[0], self.xLength)
        y=np.linspace(0, size[1], self.yLength)
        self.x, self.y=np.meshgrid(x,y)
        self.surface=np.zeros((seconds, self.xLength, self.yLength))#[[0 for x in range(int(size[0]/facet))] for y in range(int(size[1]/facet))] 
        self.omega=np.sqrt(g*self.wavenumbers)
        self.A=np.sqrt(2*self.spectrum.reshape(-1,1)*self.spreading*self.delta_k.reshape(-1,1)*self.delta_theta)
        self.seconds=np.linspace(0, seconds-1, seconds, dtype="int")
    
    def amplitude(self, i, j):
        return np.sqrt(2*self.spectrum[i]*self.spreading[j]*self.delta_k*self.delta_theta)
    
    def frame(self):
        print(f"Generating frame 1")
        for i_index, i in enumerate(self.wavenumbers):
            for j_index, j in enumerate(self.theta):
                self.surface[0,:,:]+=self.A[i_index][j_index]*np.cos(i*(self.x*self.cosTheta[j_index]+self.y*self.sinTheta[j_index])-self.epsilon[i_index][j_index])

    def video(self):
        surfaceFrequency=fft2(self.surface[0,:,:])
        for t in self.seconds[1:]:
            print(f"Generating frame {t+1}")
            tmp=surfaceFrequency.copy()
            for i_index, i in enumerate(self.wavenumbers):
                tmp*=np.exp(1j*np.sqrt(9.8067*i)*t)
            self.surface[t,:,:]=ifft2(tmp).real()

    def generate(self):
        self.frame()
        #self.video()
        return self.surface