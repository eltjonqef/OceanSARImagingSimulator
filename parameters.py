from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
class parameters:
    def __init__(self) -> None:
        self.MONOCHROMATIC=False
        self.spectrum=spectrum_model.Elfouhaily
        self.spreading=spreading_model.Longuet_Higgins
        self.n=6
        self.S=12
        self.length=2560
        self.N=128
        self.wind_speed=20
        self.wind_direction=np.pi/4
        self.seconds=5
        self.timestep=0.5
        self.fetch=25000
        self.elfouhaily_k=0.1
        self.incidence_angle=np.deg2rad(23.5)
        self.range_resolution=5
        self.azimuth_resolution=5
        self.frequency=5300000000
        self.beta=None