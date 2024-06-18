from omnidirectional_spectrum import spectrum_model
from spreading_function import spreading_model
import numpy as np
from frequencyEnums import Polarization
import yaml
class parameters:
    def __init__(self, param_file) -> None:
        CONFIG_FILE=param_file
        with open(CONFIG_FILE, "r") as f:
            self.params=yaml.safe_load(f)
        self.validate()
        self.load()

    def validate(self):
        validate_fields=["spectrum", "spreading","polarization"]
        for field in validate_fields:
            chosen=self.params[field]['chosen']
            options=self.params[field]['options']
            if chosen not in options:
                raise Exception("Model name invalid!")
    
    def load(self):
        self.MONOCHROMATIC=False
        if self.params['spectrum']['chosen']=='Pierson_Moskowitz':
            self.spectrum=spectrum_model.Pierson_Moskowitz
        elif self.params['spectrum']['chosen']=='JONSWAP':
            self.spectrum=spectrum_model.JONSWAP
        else:
            self.spectrum=spectrum_model.Elfouhaily
        if self.params['spreading']['chosen']=='Simple_Cosine':
            self.spreading=spreading_model.Simple_Cosine
        elif self.params['spreading']['chosen']=='Longuet_Higgins':
            self.spreading=spreading_model.Longuet_Higgins
        else:
            self.spreading=spreading_model.Elfouhaily
        if self.params['polarization']['chosen']=='Vertical':
            self.polarization=Polarization.Vertical
        else:
            self.polarization=Polarization.Horizontal
        self.n=self.params["n"]
        self.S=self.params["S"]
        self.length_x=self.params["length_x"]
        self.length_y=self.params["length_y"]
        self.N_x=self.params["N_x"]
        self.N_y=self.params["N_y"]
        self.wind_speed=self.params["wind_speed"]
        self.wind_direction=np.deg2rad(self.params["wind_direction"])
        self.seconds=5
        self.timestep=0.5
        self.fetch=self.params["fetch"]
        self.elfouhaily_k=self.params["elfouhaily_k"]
        self.incidence_angle=np.deg2rad(self.params["incidence_angle"])
        self.range_resolution=self.params["range_resolution"]
        self.azimuth_resolution=self.params["azimuth_resolution"]
        self.frequency=self.params["frequency"]
        self.H=self.params["H"]
        self.V=self.params["V"]
        self.beta=self.params["beta"]