import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from enum import Enum

class spreading_model(Enum):
    Simple_Cosine=0
    Longuet_Higgins=1
    Elfouhaily=2

class spreading_function:
    def __init__(self, function, theta, wind_direction, n=2, S=8, F=25000, k=0.1, v=10, good_k=None):
        self.D=np.zeros(theta.shape, dtype=np.float32)
        if good_k:
            self.theta=theta[good_k]
        else:
            self.theta=theta
        self.wind_direction=wind_direction
        self.k=k
        self.n=n
        self.S=S
        self.F=F
        self.v=v
        self.g=9.81
        self.good_k=good_k
        self.function=function
        if function==spreading_model.Simple_Cosine:
            self.D[good_k]=self.Simple_Cosine()
        elif function==spreading_model.Longuet_Higgins:
            self.D[good_k]=self.Longuet_Higgins()
        elif function==spreading_model.Elfouhaily:
            self.D[good_k]=self.Elfouhaily()

    def getSpread(self):
        return self.D
    
    def plot(self, filename):
        D=None
        if self.good_k:
            D=self.D[self.good_k]
        else:
            D=self.D
        if self.function==spreading_model.Simple_Cosine:
            plt.polar(self.theta, D, label=f"n={self.n}")
            plt.title("Simple Cosine")
        elif self.function==spreading_model.Longuet_Higgins:
            plt.polar(self.theta, D, label=f"S={self.S}")
            plt.title("Longuet Higgins")
        elif self.function==spreading_model.Elfouhaily:
            if self.k==202.68:
                label="X-band"
            elif self.k==110.23:
                label="C-band"
            elif self.k==26.73:
                label="L-band"
            else:
                label=f"k={self.k} rad/m"
            plt.polar(self.theta, D, label=label)
            plt.title("Elfouhaily")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)

    def Simple_Cosine(self):
        f=2/np.pi*np.power(np.cos(self.theta-self.wind_direction),self.n) 
        mask=(np.abs(self.theta-self.wind_direction)>np.pi/2) & (np.abs(self.theta-self.wind_direction)<3*np.pi/2)
        f[mask]=0
        return f

    def Longuet_Higgins(self):
        return (gamma(self.S+1)/(gamma(self.S+0.5)*2*np.sqrt(np.pi)))*np.cos((self.theta-self.wind_direction)/2)**(2*self.S)

    def Elfouhaily(self):
        a_0=np.log(2)/4
        a_p=4
        c_m=0.23
        k_m=370
        Cd10N=0.00144#1e-3*(0.08+0.065*v)
        uF=np.sqrt(Cd10N)*self.v
        c=np.sqrt((self.g/self.k)*(1 +np.power(self.k/k_m,2)))
        a_m=0.13*uF/c_m
        k_0=self.g/self.v**2
        X_0=2.2e-4
        X=self.g/self.v**2*self.F
        Omega_c=0.84*np.power(np.tanh(np.power(X/X_0,0.4)),-0.75)
        k_p=k_0*np.power(Omega_c,2)
        c_p=np.sqrt(self.g/k_p)
        delta_k=np.tanh(a_0+a_p*np.power(c/c_p,2.5)+a_m*np.power(c_m/c,2.5))
        return 1/(2*np.pi)*(1+delta_k*np.cos(2*(self.theta-self.wind_direction)))