### This will be my working code to create toml strings that will be parsed into the 
##rust program. 
from pydantic import BaseModel,Field
from typing import Union,Optional,List

#----------------------------------------
#----------Pulstar input-----------------
#----------------------------------------

#define pulsation mode
class Mode(BaseModel):
    l:int
    m:int
    rel_dr:float
    k:float
    frequency:float
    phase_offset:float
    rel_dtemp:float
    phase_rel_dtemp:float
    rel_dg:float
    phase_rel_dg:float
    rotation_effects:str

#define star data
class StarData(BaseModel):
    mass:float
    radius:float
    effective_temperature:float
    v_omega:float
    inclination_angle:float
    
#define phases of pulsation
class UniformTime(BaseModel):
    start:float
    end:float
    step:float

class ExplicitTime(BaseModel):
    collection:List[float]

class TimePoints(BaseModel):
    Uniform:Optional[UniformTime]=None
    Explicit:Optional[ExplicitTime]=None

#define type of meshing
class SphericalStar(BaseModel):
    theta_step:float
    phi_step:float

class Mesh(BaseModel):
    Sphere:Optional[SphericalStar]=None


class PulstarConfig(BaseModel):
    mode_data:List[Mode]
    star_data:StarData
    time_points:TimePoints
    mesh:Mesh