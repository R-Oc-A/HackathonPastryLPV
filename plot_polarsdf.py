import polars as pl
from pydantic import BaseModel,Field
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def plot_slider(df:pl.DataFrame):
    grouped=df.lazy().group_by("time").agg(pl.col("wavelength"),pl.col("normalized flux"))
    grouped_df=grouped.sort("time").collect()
    time_list=grouped_df.get_column("time").sort()


    # Setup plot
    fig, ax = plt.subplots(figsize=(10,7))
    plt.subplots_adjust(bottom=0.25)
    
    single_df=df.filter(pl.col("time") == time_list[0])

    line, = ax.plot(single_df["wavelength"].to_numpy(),single_df["normalized flux"].to_numpy(),lw=2,color='crimson')

    ax.set_xlabel("wavelength")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"time point {time_list[0]:.3f}")
    ax.grid(True)

    #Create Slider Ax
    ax_slider = plt.axes([0.2,0.1,0.6,0.03])
    time_slider = Slider(
        ax=ax_slider,
        label='Time',
        valmin=min(time_list),
        valmax=max(time_list),
        valinit=time_list[0],
        valstep=time_list)
    
    #update function
    def update(val):
        selected_time = time_slider.val
        new_df=df.filter(pl.col("time")==val)
        line.set_xdata(new_df["wave length"].to_numpy())
        line.set_ydata(new_df["normalized flux"].to_numpy())
        ax.set_title(f"time point {selected_time:.3f}")
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)

    plt.show()


def plot_grayscale(wavelength_df:pl.DataFrame):

    # Calculate the mean flux for each wavelength
    # We group by wavelength and take the mean of the Flux column
    normal_flux = wavelength_df.with_columns(pl.col("normalized flux").mean().over(pl.col("wavelength")).alias("mean"))
    delta_flux = normal_flux.select(
        [pl.col("time"),pl.col("wavelength"),pl.col("normalized flux"),
         (pl.col("normalized flux")-pl.col("mean")).alias("delta flux")]
    )
    pivot_df = delta_flux.pivot("wavelength",index="time",values="delta flux",aggregate_function="mean")#.sort("time")
    time_values = pivot_df.get_column("time").to_numpy()
    wavelength_values = wavelength_df.select(pl.col("wavelength").unique(maintain_order=True)).get_column("wavelength").to_numpy()
    grid_values = pivot_df.select(pl.exclude("time")).to_numpy()#.astype(float)
#     # Use imshow for 2D data plotting.
    # # extent defines the boundaries of the axes.
    # # aspect='auto' adjusts the aspect ratio to fill the figure.
    # # cmap='gray' sets the colormap to grayscale.
    plt.imshow(
         grid_values,
         extent=[np.min(wavelength_values), np.max(wavelength_values), np.max(time_values), np.min(time_values)],
         aspect='auto',
         cmap='gray'
     )

#     # Add plot elements
    plt.title('Flux Deviation from Mean')
    plt.xlabel('Wavelength')
    plt.ylabel('Time')
    cbar = plt.colorbar()
    cbar.set_label('Flux - Mean')

    
#     # Save the grayscale plot
    plot_name = "grayscale"
    plt.savefig(plot_name, format='jpeg', dpi=300)
    print(f"Grayscale plot saved as '{plot_name}'")

# # Display the plot
    plt.tight_layout()
    plt.show()

# 'Returns the flux intensity for a given time' 
def extract_flux(wavelength_df:pl.DataFrame,time:float=None,mean=False):
    if mean==True:
        time_list = wavelength_df.get_column("time").to_numpy
        time = time_list[0]
        normal_flux = wavelength_df.with_columns(pl.col("normalized flux").mean().over(pl.col("wavelength")).alias("mean"))
        return normal_flux.filter(pl.col("time").eq(pl.lit(time))).select("mean").to_numpy
    elif mean==False:
        return wavelength_df.filter(pl.col("time").eq(pl.lit(time))).get_column("normalized flux").to_numpy()

def plot_mean_wavelength(wavelength_df:pl.DataFrame,time:float=None,mean=False):
    pass

def extract_time_points_from_data(wavelength_df:pl.DataFrame):
    pass

def extract_wavelength_resolution_from_data(wavelength_df:pl.DataFrame):
    pass

def compute_chi_square(line_profile:pl.DataFrame,synthetic_profile:pl.DataFrame):
    pass

