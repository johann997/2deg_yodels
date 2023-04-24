import os
import base64
from io import BytesIO
import copy
import ezdxf  # pip install ezdxf https://pypi.org/project/ezdxf/

from matplotlib.path import Path
import numpy as np
import pandas as pd
import csv

import plotly
import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot, matplotlib.backends 
import matplotlib.pyplot as plt

# pio.renderers.default = "notebook_connected"

############################################################
#####              PLOTLY DEFAULT LAYOUTS              #####
############################################################
default_layout = dict(
    template="plotly_white",
    xaxis=dict(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    ),
    yaxis=dict(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    ),
)


def apply_default_layout(fig):
    """
    Applies the layout specified by the dict default_layout.

    Args:
        fig (fig): 

    Returns:
        fig: 
    """
    fig.update_layout(**default_layout)
    fig.update_xaxes(default_layout["xaxis"])
    fig.update_yaxes(default_layout["yaxis"])
    return fig


def default_fig():
    """
    Callable function to create default fig. Applies changes with apply_default_layout()

    Returns:
        fig: 
    """
    fig = go.Figure()
    apply_default_layout(fig)
    return fig


def update_plotly_layout(fig):
    fig.update_layout(
        font=dict(
            # family="Courier New, monospace",
            family="Serif",
            size=18,
            color="Black",
        )
    )


def plot_discretised_gates(
    discretised_gates,
    plot_info,
    plot_type="coordinates",
    plot=True,
    colorscale="Greens",
    color_range=None,
    plot_3d=False,
):
    """
    General plotting function, takes discretised_gates data and handles how it should be plotted. 

    Args:
        discretised_gates (dict): _description_
        plot_info (dict): _description_
        plot_type (str, optional): Which key to plot in discretised_gates. Defaults to "coordinates".
        plot (bool, optional): _description_. Defaults to True.
        colorscale (str, optional): Colorscale of heatmap. Defaults to "Greens".
        color_range (_type_, optional): Force color range of heatmap. Defaults to None which gives the range
        [np.min(z_data), np.maax(z_data)]
        plot_3d (bool, optional): Plot a 3d surface, useful for looking at potential landscape. Defaults to False.

    Returns:
        fig:
    """
    x_axis = discretised_gates["x_axis"]
    y_axis = discretised_gates["y_axis"]

    fig = default_fig()

    z_data = np.zeros((np.shape(y_axis)[0], np.shape(x_axis)[0]))
    gate_data = np.zeros((np.shape(y_axis)[0], np.shape(x_axis)[0]))

    for key, val in discretised_gates.items():
        if "val_" in key:
            z_data += val[plot_type] * val["gate_val"]
            gate_data += val["coordinates"]

    if color_range is None:
        zmin, zmax = np.min(z_data), np.max(z_data)
    else:
        zmin, zmax = color_range[0], color_range[1]

    if not plot_3d:
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_axis,
                y=y_axis,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                opacity=1,
            )
        )
    elif plot_3d:
        # 3d surface
        surface_trace = go.Surface(
            x=x_axis,
            y=y_axis,
            z=z_data,
            colorscale=colorscale,
        )
        fig.add_trace(surface_trace)

        # heatmap to see gate outline
        gate_index = np.where(gate_data == 1)
        nogate_index = np.where(gate_data != 1)
        gate_data[gate_index] = (
            gate_data[gate_index] / np.max(gate_data) * np.max(z_data)
        ) + np.max(z_data) * 0.1
        gate_data[nogate_index] = np.max(z_data) * 0.0999

        fig.add_trace(
            go.Surface(
                z=gate_data,
                x=x_axis,
                y=y_axis,
                colorscale="greys",
                showscale=False,
                reversescale=True,
                opacity=0.5,
                cmin=np.max(z_data) * 0.0999,
                cmax=np.max(z_data) * 0.1,
            )
        )

    fig.update_layout(
        #         title=f'Plotting {dxf_file}',
        yaxis_zeroline=True,
        xaxis_zeroline=True,
        #                   width = 800,
        #                   height = 800,
        #                   xaxis_range = plot_info['x_range_to_plot'],
        #                   yaxis_range = plot_info['y_range_to_plot'],
    )

    update_plotly_layout(fig)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    if plot:
        fig.show()

    return fig




def get_discretised_gates(plot_info, polyline_gates):
    """
    Create discretised_gates dict from polyline_gates

    Args:
        plot_info (dict): _description_
        polyline_gates (dict): _description_

    Returns:
        dict: discretised_gates gates format used by other functions to store info about
        the location of gates
    """
    discretised_gates = {}
    nx = plot_info["nx"]
    ny = plot_info["ny"]

    x_axis = np.linspace(np.min(plot_info["x_range"]), np.max(plot_info["x_range"]), nx)
    y_axis = np.linspace(np.min(plot_info["y_range"]), np.max(plot_info["y_range"]), ny)
    z_data = np.zeros((ny, nx))

    xx_axis, yy_axis = np.meshgrid(x_axis, y_axis)
    coors = np.hstack(
        (xx_axis.reshape(-1, 1), yy_axis.reshape(-1, 1))
    )  # coors.shape is (4000000,2)

    gate_num = 0
    for key, val in polyline_gates.items():
        x_data = val["x_array"]
        y_data = val["y_array"]

        poly_path = Path(np.stack((x_data, y_data), axis=1))
        mask = poly_path.contains_points(coors)
        mask_2d = np.reshape(mask, (-1, nx))

        x_data = xx_axis[mask_2d]
        y_data = yy_axis[mask_2d]

        if np.sum(mask_2d.astype(int)) > 0:
            gate_dict = {
                "coordinates": mask_2d.astype(int),
                "gate_val": 1,
            }
            discretised_gates[f"val_{gate_num}"] = gate_dict
            gate_num += 1
            z_data = z_data + mask_2d.astype(int)

    discretised_gates["x_axis"] = x_axis
    discretised_gates["y_axis"] = y_axis

    return discretised_gates



def get_potential_from_gate(discretised_gates, material_info):
    """
    Calculate the potential at the 2deg_depth specified in material_info using 
    Davies (1995) update discretised_gates with new key potential.

    Args:
        discretised_gates (dict): _description_
        material_info (dict): _description_

    Returns:
        dict: Updated discretised_gates dict with new key giving potential all all grid points. 
    """
    x_axis = discretised_gates["x_axis"]
    y_axis = discretised_gates["y_axis"]
    del_x_half = np.abs((x_axis[1] - x_axis[0]) / 2)
    del_y_half = np.abs((y_axis[1] - y_axis[0]) / 2)

    xx_axis, yy_axis = np.meshgrid(x_axis, y_axis)

    len_x = np.shape(x_axis)[0]
    len_y = np.shape(y_axis)[0]

    d = material_info["2deg_depth"]

    def get_g(u, v):
        R = np.sqrt(u**2 + v**2 + d**2)
        return np.arctan((u * v) / (d * R)) / (2 * np.pi)

    for key, val in discretised_gates.items():
        if "val_" in key:
            potential_data = np.zeros((len_y, len_x))

            gate_coords = val["coordinates"]

            for x_index, x in enumerate(x_axis):
                for y_index, y in enumerate(y_axis):
                    if gate_coords[y_index, x_index] != 0:
                        L = x - del_x_half
                        R = x + del_x_half
                        B = y - del_y_half
                        T = y + del_y_half

                        g1 = get_g(xx_axis - L, yy_axis - B)
                        g2 = get_g(xx_axis - L, -yy_axis + T)
                        g3 = get_g(-xx_axis + R, yy_axis - B)
                        g4 = get_g(-xx_axis + R, -yy_axis + T)
                        potential_data += g1 + g2 + g3 + g4

            discretised_gates[key]["potential"] = potential_data

    return discretised_gates



def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI, to show matplotlib figs in Dash
    https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb
    Args:
        in_fig (plt.Figure): matplotlib figure
        close_all (bool, optional): _description_. Defaults to True.

    Returns:
        str: 
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


###########################################################
#####          SAVING AND READING DATA TO CSV         #####
###########################################################
def save_geometric_potential_to_csv(
    discretised_gates, csv_name="temp_processed_data/geometric_potential.csv"
):
    """
    Save the dict discretised_gates to a .csv so data can be read back at a later time. 
    The dict contains information in 2d arrays which have to be flattened to store in .csv

    Args:
        discretised_gates (dict): 
        csv_name (str, optional): Name of .csv to save to. 
        Defaults to "temp_processed_data/geometric_potential.csv".
    """
    csv_df = pd.DataFrame([])

    gate_num = 0
    for key, val in discretised_gates.items():
        if "val_" in key:
            temp_df = pd.DataFrame(
                {
                    f"val_{gate_num}_coordinates": discretised_gates[key][
                        "coordinates"
                    ].flatten(),
                    f"val_{gate_num}_potential": discretised_gates[key][
                        "potential"
                    ].flatten(),
                }
            )
            csv_df = pd.concat([csv_df, temp_df], axis=1)
            gate_num += 1
        else:
            temp_df = pd.DataFrame({key: discretised_gates[key].flatten()})
            csv_df = pd.concat([csv_df, temp_df], axis=1)

    csv_df.to_csv(f"{csv_name}", index=False)


def get_1d_to_2d(array, nx, ny):
    """
    Reshape a 1d array to 2d given numpts in x and y direction
    Args:
        array (np.array()): 1d array
        nx (int): Numpts in x-direction
        ny (int): Numpts in y-direction

    Returns:
        np.array: 2d array
    """
    return np.array(array).reshape(ny, nx)


def get_discretised_gates_from_csv(
    nx, ny, csv_name="temp_processed_data/geometric_potential.csv"
):
    """
    Read .csv and return a dict containing coordinates of gates.
    The data is 2d but flattened before saved to .csv so nx and ny are used to reshape the data.

    Args:
        nx (int): Numpts in x-direction
        ny (int): Numpts in y-direction
        csv_name (str, optional): Name of .csv containing info. 
        Defaults to "temp_processed_data/geometric_potential.csv".

    Returns:
        dict: discretised_gates
    """
    df = pd.read_csv(csv_name)
    column_names = df.columns

    discretised_gates = {}
    gate = {}

    old_val = ""
    for index, column_name in enumerate(df):
        val = "".join(column_name.split("_")[0:2])

        if "val" in val:
            gate[column_name.split("_")[-1]] = get_1d_to_2d(df[column_name], nx, ny)
            try:
                if "".join(column_names[index + 1].split("_")[0:2]) != val:
                    discretised_gates["_".join(column_name.split("_")[0:2])] = gate
                    gate = {}
            except:
                continue
        else:
            discretised_gates[column_name] = df[column_name].dropna().to_numpy()

    return copy.deepcopy(discretised_gates)



def get_plot_info(depth_2deg, minx, maxx, miny, maxy, nx, ny):
    max_range = np.max([maxy - miny, maxx - minx])
    mean_x = (minx + maxx) / 2
    mean_y = (miny + maxy) / 2

    plot_info = {
        "x_range": [minx, maxx],
        "y_range": [miny, maxy],
        "max_range": max_range,
        "x_range_to_plot": np.array([-max_range, max_range]) / 2 + mean_x,
        "y_range_to_plot": np.array([-max_range, max_range]) / 2 + mean_y,
        "nx": nx,
        "ny": ny,
        "2deg_depth": depth_2deg,
    }

    return plot_info


def save_file(name, content, upload_directory):
    """
    Decode and store a file uploaded with Plotly Dash."

    Args:
        name (str): name of .dxf contiining polyline data
        content (data): data read by Dash
        upload_directory (str): ile path to .dxf
    """
    if name is not None:
        data = content.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(upload_directory, name), "wb") as fp:
            fp.write(base64.decodebytes(data))


############################################################
#####                  DXF TO POLYLINE                 #####
############################################################
def save_dxf_to_csv(name, upload_directory):
    """
    Reead a .dxf with 'name' at file path upload_directory. Create polylines from 
    .dxf and save to .csv every two columns represent a seperate gate in the .dxf
    even columns are x-coordinates and odd columns are y-coordinates.

    Args:
        name (str): name of .dxf contiining polyline data
        upload_directory (sr): file path to .dxf
    """
    file_path = f"{upload_directory}/{name}"
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    polyline_points = {}
    # entity query for all POLYLINE entities in modelspace
    # https://ezdxf.readthedocs.io/en/stable/dxfentities/polyline.html
    for index, e in enumerate(msp.query("POLYLINE")):
        val = f"val_{index}"
        points = np.array(list(e.points()))
        polyline_points[val] = np.vstack((points, points[0]))

    csv_df = pd.DataFrame([])

    for key, val in polyline_points.items():
        temp_df = pd.DataFrame({f"{key}_xcoord": val[:, 0], f"{key}_ycoord": val[:, 1]})
        csv_df = pd.concat([csv_df, temp_df], axis=1)

    csv_df.to_csv(f"{upload_directory}/{name.split('.')[0]}.csv", index=False)


def read_csv_to_polyline(name, upload_directory):
    """
    Read a dict of polyline data read from .csv with 'name' at file path
    'upload_directory'
    Args:
        name (str): name of .csv contiining polyline data
        upload_directory (str): file path to .csv

    Returns:
        dict: each key represents a different gate each gate contains a sub
        dict with key x_array and y_array which are the coordinates of the polyline points
    """
    csv_name = f"{upload_directory}/{name.split('.')[0]}.csv"
    df = pd.read_csv(csv_name)
    column_names = df.columns
    polyline_gates = {}

    for index in range(int(len(column_names) / 2 - 1)):
        val = "".join(column_names[int(2 * index)].split("_")[0:2])

        x_array = df[column_names[int(2 * index)]].dropna().to_numpy()
        y_array = df[column_names[int(2 * index + 1)]].dropna().to_numpy()

        gate = {
            "x_array": x_array,
            "y_array": y_array,
        }

        polyline_gates[val] = gate
        gate = {}

    return polyline_gates


def plot_polyline(name, upload_directory, dxf_file=""):
    """
    Plots a figure using the polyline data stored in the .csv 'name'

    Args:
        name (str): name of .csv contiining polyline data
        upload_directory (str): file path to .csv
        dxf_file (str, optional): Used to add name to plot title. Defaults to "".

    Returns:
        fig: 
    """
    polyline_gates = read_csv_to_polyline(name, upload_directory)

    fig = default_fig()

    for key, val in polyline_gates.items():
        x_data = val["x_array"]
        y_data = val["y_array"]

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                marker_line_width=3,
                name=f"{key}",
                # fill="toself"
            )
        )

    fig.update_layout(
        title=f"Plotting {dxf_file}",
        yaxis_zeroline=True,
        xaxis_zeroline=True,
    )

    update_plotly_layout(fig)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def uploaded_files(upload_directory):
    """
    List the files in the upload directory.

    Args:
        upload_directory (str): File path of temp folder used to store data in app

    Returns:
        _type_: _description_
    """
    files = []
    for filename in os.listdir(upload_directory):
        path = os.path.join(upload_directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files



############################
#####       MATH       #####
############################

def reduce_multiple(array):
    """
    Returns an array of ints divided by greatest common divider
    e.g.
    np.array([10,20,30]) -> np.array([1,2,3])

    Args:
        array (np.array): 

    Returns:
        np.array: _description_
    """
    return (array / np.gcd.reduce(array)).astype(int)