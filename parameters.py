from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from frequencyEnums import Polarization
class parameters:
    def __init__(self) -> None:
        self.MONOCHROMATIC=False
        self.spectrum=spectrum_model.Elfouhaily
        self.spreading=spreading_model.Longuet_Higgins
        self.polarization=Polarization.Vertical
        self.n=6
        self.S=12
        self.length_x=1000
        self.length_y=1000
        self.N_x=512
        self.N_y=512
        self.wind_speed=10
        self.wind_direction=np.pi/4
        self.seconds=5
        self.timestep=0.5
        self.fetch=200000
        self.elfouhaily_k=0.1
        self.incidence_angle=np.deg2rad(30)
        self.range_resolution=5
        self.azimuth_resolution=5
        self.frequency=5000000000
        self.beta=50