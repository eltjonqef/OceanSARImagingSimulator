import numpy as np
from scipy.special import gamma
def Simple_Cosine(theta,n):
    return 2/np.pi*np.power(np.cos(theta),n) 

def Longuet_Higgins(theta, S):
    return (gamma(S+1)/(gamma(S+0.5)*2*np.sqrt(np.pi)))*np.power(np.cos(theta/2),2*S)

def Elfouhaily(theta, k, v):
    g=9.80665
    F=25000
    a_0=np.log(2)/4
    a_p=4
    c_m=0.23
    k_m=370
    Cd10N=0.00144#1e-3*(0.08+0.065*v)
    uF=np.sqrt(Cd10N)*v
    c=np.sqrt((g/k)*(1 +np.power(k/k_m,2)))
    a_m=0.13*uF/c_m
    k_0=g/v**2
    X_0=2.2e-4
    X=g/v**2*F
    Omega_c=0.84*np.power(np.tanh(np.power(X/X_0,0.4)),-0.75)
    k_p=k_0*np.power(Omega_c,2)
    c_p=np.sqrt(g/k_p)
    delta_k=np.tanh(a_0+a_p*np.power(c/c_p,2.5)+a_m*np.power(c_m/c,2.5))
    return 1/(2*np.pi)*(1+delta_k*np.cos(2*theta))