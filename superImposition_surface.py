import numpy as np
from omnidirectional_spectrum import omnidirectional_spectrum, spectrum_model
from spreading_function import spreading_function, spreading_model

class superImpositionSurfaceGenerator:
    def __init__(self, spectrum, spreading, length, N, wind_speed, wind_direction, n, S, seconds, timestep, fetch, elfouhaily_k):
        self.spectrum=spectrum
        self.spreading=spreading
        self.omnidirectional_spectrum=None
        self.spreading_function=None
        self.g=9.81
        self.dx=length/N
        self.L=length
        self.N=N
        self.elfouhaily_k=elfouhaily_k
        print(self.N)
        print(self.dx)
        print(self.L)
        x=np.linspace(-self.L/2,self.L/2,self.N)
        y=np.linspace(-self.L/2,self.L/2,self.N)
        self.x, self.y=np.meshgrid(x, y)
        self.time=np.linspace(0,seconds, int(seconds/timestep))
        self.surface=np.zeros((self.time.size, self.N, self.N))
        self.wind_speed=wind_speed
        self.wind_direction=wind_direction
        self.fetch=fetch
        self.wavenumbers=np.logspace(-3,5,500)
        delta_k=np.diff(self.wavenumbers,prepend=self.wavenumbers[0])
        self.theta=np.linspace(0,2*np.pi, 200)
        delta_theta=np.diff(self.theta, prepend=self.theta[0])
        self.omnidirectional_spectrum=omnidirectional_spectrum(spectrum=self.spectrum,k=self.wavenumbers, v=self.wind_speed,F=self.fetch)
        self.spreading_function=spreading_function(function=self.spreading, theta=self.theta, n=n, S=S, F=self.fetch, k=self.elfouhaily_k, v=self.wind_speed)
        self.S=self.omnidirectional_spectrum.getSpectrum()
        self.D=self.spreading_function.getSpread()
        self.PSI=self.S.reshape(-1,1)*self.D
        self.kPSI=(self.wavenumbers**2*self.S).reshape(-1,1)*self.D
        self.A=np.sqrt(2*self.PSI*delta_k.reshape(-1,1)*delta_theta)
        self.omega=np.sqrt(self.g*self.wavenumbers)
        self.epsilon=np.random.uniform(0,2*np.pi,(len(self.wavenumbers),len(self.theta)))
    
    def generate(self):
        for frame, t in enumerate(self.time):
            for i, k in enumerate(self.wavenumbers):
                for j, theta in enumerate(self.theta):
                    self.surface[frame,:,:]+=self.A[i][j]*np.cos(k*(self.x*np.cos(theta+self.wind_direction)+self.y*np.sin(theta+self.wind_direction))-self.omega[i]*t+self.epsilon[i][j])
                print(i)
        return self.surface
    
    def orbital_velocity(self, incidence_angle):
        u_x=np.zeros((self.time.size, self.N, self.N))
        u_y=np.zeros((self.time.size, self.N, self.N))
        u_z=np.zeros((self.time.size, self.N, self.N))
        for i, k in enumerate(self.wavenumbers):
            for j, theta in enumerate(self.theta):
                u_x+=self.A[i][j]/self.omega[i]*k*np.cos(k*(self.x*np.cos(theta+self.wind_direction)+self.y*np.sin(theta+self.wind_direction))+self.epsilon[i][j])*np.cos(theta+self.wind_direction)
                u_y+=self.A[i][j]/self.omega[i]*k*np.cos(k*(self.x*np.cos(theta+self.wind_direction)+self.y*np.sin(theta+self.wind_direction))+self.epsilon[i][j])*np.sin(theta+self.wind_direction)
                u_z+=self.A[i][j]/self.omega[i]*k*np.sin(k*(self.x*np.cos(theta+self.wind_direction)+self.y*np.sin(theta+self.wind_direction))+self.epsilon[i][j])
            print(i)

        u_x=u_x*9.81
        u_y=u_y*9.81
        u_z=u_z*9.81
        u_r=u_z*np.cos(incidence_angle)-np.sin(incidence_angle)*(u_x*np.sin(self.wind_direction)+u_y*np.cos(self.wind_direction))
        return u_r
    
    def getSurfaceVariances(self):
        return [np.var(self.surface[frame,:,:]) for frame, _ in enumerate(self.time)]
    
    def getSpectrumIntegral(self):
        return np.trapz(np.trapz(self.PSI,self.wavenumbers, axis=0), self.theta, axis=0)
    
    def getSlopes(self):
        Sx=np.gradient(self.surface[0,:,:],axis=0).flatten()
        Sy=np.gradient(self.surface[0,:,:],axis=1).flatten()
        return np.var(Sx)+np.var(Sy)

    def getSlopeIntegral(self):
        return np.trapz(np.trapz(self.kPSI,self.wavenumbers, axis=0), self.theta, axis=0)
    
    def getSignificantWaveHeights(self):
        return [4*np.sqrt(np.var(self.surface[frame,:,:])) for frame, _ in enumerate(self.time)]
