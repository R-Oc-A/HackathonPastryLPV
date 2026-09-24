
from pydantic import BaseModel,Field
from typing import Union,Optional,List
#----------------------------------------
#----------Profile input-----------------
#----------------------------------------

#Intensity grid
class JorisGrid(BaseModel):
    temperature:float
    log_gravity:float
    filename:str


class NadyaGrid(BaseModel):
    temperature:float
    log_gravity:float
    metalicity:float
    filename:str

class EmaGrid(BaseModel):
    temperature:float
    log_gravity:float
    metalicity:float
    filename:str

class IntensityGrid(BaseModel):
    Joris:Optional[JorisGrid]=None
    Nadya:Optional[NadyaGrid]=None
    EmaParquet:Optional[EmaGrid]=None

#Wave length range
class WavelengthRange(BaseModel):
    start:float
    end:float
    step:float

#profile_config
class ProfileConfig(BaseModel):
    max_velocity:float
    path_to_grids:str
    wavelength_range:WavelengthRange
    intensity_grids:List[IntensityGrid]

#Gaussian Profile config
class GaussianProfile(BaseModel):
    sigma:float
    eq_w:float
    alpha_w:float
    zero_point_shift:float = 0.0
    central_wavelength:float
    left_wavelength:float
    right_wavelength:float
    step:float
    t_eff:float
    mass:float
    radius:float