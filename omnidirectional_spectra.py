import numpy as np

def pierson_moskowitz(k, v):
    a=0.0081
    b=0.74
    g=9.8067
    V4=v**4
    return (a/(2*np.power(k,3)))*np.exp(-b*np.power(g/k,2)/V4)
    
def JONSWAP(k, v):
    F=80000
    g=9.8067
    gamma=3.3
    a=0.076*(v**2/(F*g))**0.22
    k_p=(7*np.pi*(g/(v*np.sqrt(g)))*np.power(v**2/(g*F),0.33))**2
    k_pIndex=np.searchsorted(k, k_p)
    k_low=k[:k_pIndex]
    k_high=k[k_pIndex:]
    sigma_low=0.07
    sigma_high=0.09
    S_low=np.array([])
    S_high=np.array([])
    if np.size(k_low):
        S_low=a/2*np.power(k_low,-3)*np.exp(-1.25*np.power(k_low/k_p,-2))*np.exp(np.log(gamma)*np.exp(-np.power(np.sqrt(k_low/k_p)-1,2)/(2*np.power(sigma_low,2))))
    if np.size(k_high):
        S_high=a/2*np.power(k_high,-3)*np.exp(-1.25*np.power(k_high/k_p,-2))*np.exp(np.log(gamma)*np.exp(-np.power(np.sqrt(k_high/k_p)-1,2)/(2*np.power(sigma_high,2))))    
    return np.concatenate((S_low, S_high))

def lower(k, v):
    F=80000
    g=9.80665
    k_0=g/v**2
    X_0=2.2e-4
    X=g/v**2*F
    Omega_c=0.84*np.power(np.tanh(np.power(X/X_0,0.4)),-0.75)
    k_p=k_0*np.power(Omega_c,2)
    c_p=np.sqrt(g/k_p)
    Omega=v/c_p
    a_p=6e-3*np.sqrt(Omega)
    L_pm=np.exp(-1.25*np.power(k_p/k,2))
    sigma=0.08*(1+4*np.power(Omega_c,-3))
    Gamma=np.exp(-np.power(np.sqrt(k/k_p)-1,2)/(2*np.power(sigma,2)))
    if Omega_c<=1:
        gamma=1.7
    else:
        gamma=1.7+6*np.log(Omega_c)
    J_p=np.power(gamma,Gamma)
    k_m=370
    F_p=L_pm*J_p*np.exp(-Omega/np.sqrt(10)*(np.sqrt(k/k_p)-1))
    c=np.sqrt((g/k)*(1 +np.power(k/k_m,2))) # i am not sure about this
    return 1/2*a_p*(c_p/c)*F_p,L_pm,J_p



def higher(k,v,L_pm,J_p):
    g=9.80665
    c_m=0.23
    a_0=1.4e-2
    Cd10N=0.00144#1e-3*(0.08+0.065*v)
    uF=np.sqrt(Cd10N)*v
    a_m=0
    if uF<=c_m:
        a_m=1e-2*(1+np.log(uF/c_m))
    else:
        a_m=1e-2*(1+3*np.log(uF/c_m))
    #a_m=a_0*uF/c_m
    k_m=370
    #L_pm=np.exp(-1.25*np.power(k_m/k,2))
    F_m=L_pm*J_p*np.exp(-1/4*np.power(k/k_m-1,2))
    c=np.sqrt((g/k)*(1 +np.power(k/k_m,2)))  # i am not sure about this
    return 1/2*a_m*(c_m/c)*F_m

def Elfouhaily(k,v):
    B_l,L_pm,J_p=lower(k,v)
    B_h=higher(k,v,L_pm,J_p)
    return np.power(k,-3)*(B_l+B_h)