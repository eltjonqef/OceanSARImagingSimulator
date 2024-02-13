import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class spectrum_model(Enum):
    Pierson_Moskowitz=0
    JONSWAP=1
    Elfouhaily=2

class omnidirectional_spectrum:
    def __init__(self, spectrum, k, v, F=25000, good_k=None):
        self.S=np.zeros(k.shape, dtype=np.float32)
        self.F=F
        self.k=k[good_k]
        self.v=v
        self.g=9.81
        self.spectrum=spectrum
        if spectrum==spectrum_model.Pierson_Moskowitz:
            self.S[good_k]=self.pierson_moskowitz()
        elif spectrum==spectrum_model.JONSWAP:
            self.S[good_k]=self.JONSWAP()
        elif spectrum==spectrum_model.Elfouhaily:
            self.S[good_k]=self.Elfouhaily()

    def getSpectrum(self):
        return self.S
    
    def plot(self):
        plt.plot(self.k, self.S,label=f"{self.v} m/s")
        if self.spectrum==spectrum_model.Pierson_Moskowitz:
            plt.title("Pierson Moskowitz")
        elif self.spectrum==spectrum_model.JONSWAP:
            plt.title("JONSWAP")
        elif self.spectrum==spectrum_model.Elfouhaily:
            plt.title("Elfouhaily")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-15, 1e3)
        plt.xlabel('Wavenumber [rad/m]')
        plt.ylabel('Elevation Spectrum [m²/rad/m]')
        plt.legend()
        plt.grid(True)

    def pierson_moskowitz(self):
        a=0.0081
        b=0.74
        g=9.81
        V4=self.v**4
        return (a/(2*np.power(self.k,3)))*np.exp(-b*np.power(g/self.k,2)/V4)
    
    def JONSWAP(self):
        gamma=3.3
        a=0.076*(self.v**2/(self.F*self.g))**0.22
        k_p=(7*np.pi*(self.g/(self.v*np.sqrt(self.g)))*np.power(self.v**2/(self.g*self.F),0.33))**2
        k_pIndex=np.searchsorted(self.k, k_p)
        k_low=self.k[:k_pIndex]
        k_high=self.k[k_pIndex:]
        sigma_low=0.07
        sigma_high=0.09
        S_low=np.array([])
        S_high=np.array([])
        if np.size(k_low):
            S_low=a/2*np.power(k_low,-3)*np.exp(-1.25*np.power(k_low/k_p,-2))*np.exp(np.log(gamma)*np.exp(-np.power(np.sqrt(k_low/k_p)-1,2)/(2*np.power(sigma_low,2))))
        if np.size(k_high):
            S_high=a/2*np.power(k_high,-3)*np.exp(-1.25*np.power(k_high/k_p,-2))*np.exp(np.log(gamma)*np.exp(-np.power(np.sqrt(k_high/k_p)-1,2)/(2*np.power(sigma_high,2))))    
        return np.concatenate((S_low, S_high))

    def lower(self):
        k_0=self.g/self.v**2
        X_0=2.2e-4
        X=self.g/self.v**2*self.F
        Omega_c=0.84*np.power(np.tanh(np.power(X/X_0,0.4)),-0.75)
        k_p=k_0*np.power(Omega_c,2)
        c_p=np.sqrt(self.g/k_p)
        Omega=self.v/c_p
        a_p=6e-3*np.sqrt(Omega)
        L_pm=np.exp(-1.25*np.power(k_p/self.k,2))
        sigma=0.08*(1+4*np.power(Omega_c,-3))
        Gamma=np.exp(-np.power(np.sqrt(self.k/k_p)-1,2)/(2*np.power(sigma,2)))
        if Omega_c<=1:
            gamma=1.7
        else:
            gamma=1.7+6*np.log(Omega_c)
        J_p=np.power(gamma,Gamma)
        k_m=370
        F_p=L_pm*J_p*np.exp(-Omega/np.sqrt(10)*(np.sqrt(self.k/k_p)-1))
        c=np.sqrt((self.g/self.k)*(1 +np.power(self.k/k_m,2))) # i am not sure about this
        return 1/2*a_p*(c_p/c)*F_p,L_pm,J_p

    def higher(self, L_pm, J_p):
        c_m=0.23
        a_0=1.4e-2
        Cd10N=0.00144#1e-3*(0.08+0.065*v)
        uF=np.sqrt(Cd10N)*self.v
        a_m=0
        if uF<=c_m:
            a_m=1e-2*(1+np.log(uF/c_m))
        else:
            a_m=1e-2*(1+3*np.log(uF/c_m))
        #a_m=a_0*uF/c_m
        k_m=370
        #L_pm=np.exp(-1.25*np.power(k_m/k,2))
        F_m=L_pm*J_p*np.exp(-1/4*np.power(self.k/k_m-1,2))
        c=np.sqrt((self.g/self.k)*(1 +np.power(self.k/k_m,2)))  # i am not sure about this
        return 1/2*a_m*(c_m/c)*F_m

    def Elfouhaily(self):
        B_l,L_pm,J_p=self.lower()
        B_h=self.higher(L_pm,J_p)
        return np.power(self.k,-3)*(B_l+B_h)