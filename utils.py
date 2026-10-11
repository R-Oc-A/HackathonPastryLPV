import polars as pl
from huggingface_hub import hf_hub_download
import line_profile_description as lpd
import numpy as np
import tomli_w

temp_array=np.concatenate((
    np.arange(6500,10100,100,dtype=float),
    np.arange(10250,27750,250,dtype=float),
    np.arange(27750,30500,250,dtype=float),
    np.arange(30500,33500,500,dtype=float),
    ))
logg_array=np.arange(3.0,4.6,0.1,dtype=float)

def make_grid_name(temp,logg):
    temp_int=int(temp)
    logg_int= int(10*logg)
    return f"lp0000_{temp_int:05d}_{logg_int:03d}0_0020.parquet"


def download_grids(grid_names):
    """
    This function is used to download the grids that constrain the local variability
    """
    for grid_name in grid_names:
        try:
            hf_hub_download(repo_id="Ricard0draciR/SolarMetalicityGrid",
                filename=grid_name,
                repo_type="dataset",
                local_dir="content/grids/SolarMetalicity"
                )
        except:
            raise FileNotFoundError("unable to download \n{grid_name}\n from repo")
    


def define_parameter_space_grids(
        temp_min,
        temp_max,
        logg_min,
        logg_max):
    """
    This function defines the four points on the parameter space grids where you may find the grids
    Constructs the logical definitions usable for the profiler program 
    and downloads the requested model grids if they are not locally in the machine already.
    ### Arguments:
    * temp_min: Minimum value of the effective temperature of the pulsation
    * temp_max: Maximum value of the effective temperature of the pulsation
    * logg_min: Minimum value of the logarithm of the surface gravity of the pulsation
    * logg_max: Maximum value of the logarithm of the surface gravity of the pulsation
    ### Returns:
    * This function returns the points in the parameter space that constraint the pulsation of a star
    """
    (grid_minT,grid_maxT,grid_minlogg,grid_maxlogg)=get_coordinates_parameter_space(temp_min,temp_max,logg_min,logg_max)
    grid_names=[
        make_grid_name(grid_minT,grid_minlogg),
        make_grid_name(grid_minT,grid_maxlogg),
        make_grid_name(grid_maxT,grid_minlogg),
        make_grid_name(grid_maxT,grid_maxlogg),
    ]

    download_grids(grid_names)

    grid1 = lpd.IntensityGrid(
        EmaParquet=lpd.EmaGrid(temperature=grid_minT,
                               log_gravity=grid_minlogg,
                               metalicity=0.0,
                               filename=grid_names[0]))
    grid2 = lpd.IntensityGrid(
        EmaParquet=lpd.EmaGrid(temperature=grid_minT,
                               log_gravity=grid_maxlogg,
                               metalicity=0.0,
                               filename=grid_names[1]))
    grid3 = lpd.IntensityGrid(
        EmaParquet=lpd.EmaGrid(temperature=grid_maxT,
                               log_gravity=grid_minlogg,
                               metalicity=0.0,
                               filename=grid_names[2]))
    grid4 = lpd.IntensityGrid(
        EmaParquet=lpd.EmaGrid(temperature=grid_maxT,
                               log_gravity=grid_maxlogg,
                               metalicity=0.0,
                               filename=grid_names[3]))
    return(grid1,
        grid2, 
        grid3,
        grid4)
                    

def get_coordinates_parameter_space(
        temp_min,
        temp_max,
        logg_min,
        logg_max):
    """
    This function gives you the coordinates of the closest points that define a hypercube of the parameter space that constrain the values of pulsation
    ### Arguments:
    * temp_min: Minimum value of the effective temperature of the pulsation
    * temp_max: Maximum value of the effective temperature of the pulsation
    * logg_min: Minimum value of the logarithm of the surface gravity of the pulsation
    * logg_max: Maximum value of the logarithm of the surface gravity of the pulsation
    ### Returns: 
    * This function returns a tupple with
    (minT,maxT,minlogg,maxlogg)
    """
    indx_max_temp=np.argmin(np.abs(temp_array-temp_max))
    indx_min_temp=np.argmin(np.abs(temp_array-temp_min))
    indx_max_logg=np.argmin(np.abs(logg_array-logg_max))
    indx_min_logg=np.argmin(np.abs(logg_array-logg_min))

    max_modelled_temp = temp_array[indx_max_temp]
    max_modelled_logg = logg_array[indx_max_logg]
    min_modelled_temp = temp_array[indx_min_temp]
    min_modelled_logg = logg_array[indx_min_logg]
    
    errorstring = "Pulsation local values are outside of the ones precomputed in the modelled grids"
    if temp_max>max_modelled_temp:
        try:
            max_modelled_temp=temp_array[indx_max_temp+1]
        except:
            raise ValueError(f"Maximum value of temperature is {temp_max}\n"+errorstring)
        else:
            if logg_max>max_modelled_logg:
                try:
                    max_modelled_logg=logg_array[indx_max_logg+1]
                except:
                    raise ValueError(f"Maximum value of logg is {logg_max}\n" + errorstring)
    if temp_min<min_modelled_temp:
        min_modelled_temp=temp_array[indx_min_temp-1]
        if temp_min<min_modelled_temp:
            raise ValueError(f"Minimum value of temperature is {temp_min}\n"+errorstring)
    if logg_min<min_modelled_logg:
        min_modelled_logg=logg_array[indx_min_logg-1]
        if logg_min<min_modelled_logg:
            raise ValueError(f"Minimum value of logg is {logg_min}\n"+errorstring)
    
    return (min_modelled_temp,
            max_modelled_temp,
            np.trunc(min_modelled_logg*10)/10,
            np.trunc(max_modelled_logg*10)/10)



def extract_extremal_local_values_from_star(star):
    max_temp=star.select(pl.max("temperature")).to_numpy()[0]
    min_temp=star.select(pl.min("temperature")).to_numpy()[0]
    max_logg=star.select(pl.max("log gravity")).to_numpy()[0]
    min_logg=star.select(pl.min("log gravity")).to_numpy()[0]
    return(min_temp[0],max_temp[0],min_logg[0],max_logg[0])

def make_profile_input(star,wavelength_range):
    ### Download and parse de intensity grids
    (temp_min,temp_max,logg_min,logg_max)=extract_extremal_local_values_from_star(star)
    print("Downloading and parsing modelled grids:")
    (grid1,grid2,grid3,grid4)=define_parameter_space_grids(temp_min,temp_max,logg_min,logg_max)
    path_to_grids = f"content/grids/SolarMetalicity/"

    prof_config=lpd.ProfileConfig(
        max_velocity=1.0e2,
        path_to_grids=path_to_grids,
        wavelength_range=wavelength_range,
        intensity_grids=[grid1,grid2,grid3,grid4]
    )
    prof_config_dict=prof_config.model_dump(exclude_none=True)
    prof_config_string=tomli_w.dumps(prof_config_dict)
    return prof_config_string



# hf_hub_download(repo_id="Ricard0draciR/SolarMetalicityGrid",
#                 filename="lp000_30500_0350_0020.parquet",
#                 repo_type="dataset",
#                 local_dir="content/grids/SolarMetalicity"
#                 )