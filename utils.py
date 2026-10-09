import polars as pl
from huggingface_hub import hf_hub_download
import line_profile_description

'''
This function is used to download the grids that constrain the local variability
'''
def download_grids(minTemp,maxTemp,minLogG,maxLogG):
    pass


'''
This function defines the four points on the parameter space grids where you may find the grids
'''
def define_parameter_space_grids(min):
    pass

'''
This function is used to get the maximum and minimum local values of log_g and Temp 
from a pulsating star simulated using pastrypy
'''
def get_max_values(star):
    pass

def get_profile_input(star):
    pass


# hf_hub_download(repo_id="Ricard0draciR/SolarMetalicityGrid",
#                 filename="lp000_30500_0350_0020.parquet",
#                 repo_type="dataset",
#                 local_dir="content/grids/SolarMetalicity"
#                 )