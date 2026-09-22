import polars as pl
from pydantic import BaseModel,Field
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider




def plot_slider(df:pl.DataFrame):
    grouped=df.lazy().group_by("time").agg(pl.col("wave length"),pl.col("normalized flux"))
    grouped_df=grouped.sort("time").collect()
    time_list=grouped_df.get_column("time").sort()


    # Setup plot
    fig, ax = plt.subplots(figsize=(10,7))
    plt.subplots_adjust(bottom=0.25)
    
    single_df=df.filter(pl.col("time") == time_list[0])

    line, = ax.plot(single_df["wave length"].to_numpy(),single_df["normalized flux"].to_numpy(),lw=2,color='crimson')

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


