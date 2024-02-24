import numpy as np

class SAR_imaging:
    def __init__(self, surface, PSI, look_angle, k, omega ):
        self.surface=surface
        self.look_angle=look_angle
        self.k=k[0,:]
        self.theta=np.linspace(0, 2*np.pi, len(k))
        self.PSI=PSI
        self.omega=omega[0,:]
        self.mu=0.5

    def normalized_radar_cross_section(self):
        pass

    # def NRCS(self):
    #     return 8*np.pi*ke*np.cos()**4*self.PSI

    def complex_scattering(self):
        

    def tilt(self):
        return -(4/np.tan(self.look_angle))/(1+np.sin(self.look_angle)**2)*(1j*self.k*np.sin(self.theta))
    
    def hydrodynamic(self):
        return 4.5*self.omega*(self.k*np.sin(self.theta))**2*(self.omega-self.mu*1j)/(np.abs(self.k)*(self.omega**2+self.mu**2))
    
