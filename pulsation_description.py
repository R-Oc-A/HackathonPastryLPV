### This will be my working code to create toml strings that will be parsed into the 
##rust program. 
from pydantic import BaseModel,Field
from typing import Union,Optional,List
import numpy as np

#----------------------------------------
#----------Pulstar input-----------------
#----------------------------------------

class NonRot(BaseModel):pass
class PerturbCor(BaseModel):pass
class TAR(BaseModel): pass    
class CenDef(BaseModel):
    coefficient_expansion:List[float]

class RotationRegime(BaseModel):
    NonRotating:Optional[NonRot]= Field(default_factory=NonRot)
    PerturbativeCoriolis:Optional[PerturbCor]=None
    Tar:Optional[TAR]=None
    CentrifugalDeformation:Optional[CenDef]=None
    
#define star data
class StarData(BaseModel):
    mass:float=10.0
    radius:float=6.93
    effective_temperature:float=21642.0
    v_omega:float=20.0
    inclination_angle:float=45.0

#define pulsation mode
class Mode(BaseModel):
    l:int=2
    m:int=1
    rel_dr:float=0.024
    k:float=0.05
    frequency:float=5.38
    phase_offset:float=0.0
    rel_dtemp:float=2.62
    phase_rel_dtemp:float=180.0
    rel_dg:float=10.0
    phase_rel_dg:float=34.0
    rotation_effects:RotationRegime=RotationRegime()
    def compute_k_value(self,star:StarData):
        self.k = 74.437*star.mass/(np.power(star.radius,3)*np.power(self.frequency,2))
    
#define phases of pulsation
class UniformTime(BaseModel):
    start:float=0.0
    end:float=0.1
    step:float=0.01

class ExplicitTime(BaseModel):
    collection:List[float]=[0.0,0.01]

class TimePoints(BaseModel):
    Uniform:Optional[UniformTime]=None
    Explicit:Optional[ExplicitTime]=None

#define type of meshing
class SphericalStar(BaseModel):
    theta_step:float=2.0
    phi_step:float=4.0

class HealpixStar(BaseModel):
    depth:int=3.0

class TriangulatedStar(BaseModel):
    triangle_length:float=0.3

class DeformedStar(BaseModel):
    triangle_length:float=0.3
    rotation_frequency:float=0.33

class Mesh(BaseModel):
    Sphere:Optional[SphericalStar]=None
    HSphere:Optional[HealpixStar]=None
    TSphere:Optional[TriangulatedStar]=None
    DSphere:Optional[DeformedStar]=None


class PulstarConfig(BaseModel):
    mode_data:List[Mode]=[Mode()]
    star_data:StarData=StarData()
    time_points:TimePoints=TimePoints()
    mesh:Mesh=Mesh()


