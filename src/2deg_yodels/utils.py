import os
import base64
import ezdxf  # pip install ezdxf https://pypi.org/project/ezdxf/

from matplotlib.path import Path
import numpy as np
import pandas as pd
import csv

import plotly
import plotly.graph_objects as go
import plotly.io as pio

# pio.renderers.default = "notebook_connected"


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
    fig.update_layout(**default_layout)
    fig.update_xaxes(default_layout["xaxis"])
    fig.update_yaxes(default_layout["yaxis"])
    return fig


def default_fig():
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

        # heatmap_trace.colorbar.update(dict(outlinewidth=0, thickness=15, len=1))
        # heatmap_trace.colorbar.update_z(zmin=np.min(z_data))
        # heatmap_trace.update_z(z=np.min(z_data))

        # fig.update_layout(
        #     autosize=False,
        #     scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        #     margin=dict(l=65, r=50, b=65, t=90),
        # )

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


def get_potential_from_gate(discretised_gates, material_info):
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

            discretised_gates[key]["potential"] = potential_data * val["gate_val"]

    return discretised_gates


def save_geometric_potential_to_csv(
    discretised_gates, csv_name="temp_processed_data/geometric_potential.csv"
):
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
    return np.array(array).reshape(ny, nx)


def get_discretised_gates_from_csv(
    nx, ny, csv_name="temp_processed_data/geometric_potential.csv"
):
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

    return discretised_gates


def get_discretised_gates(plot_info, polyline_gates):
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
    """Decode and store a file uploaded with Plotly Dash."""
    if name is not None:
        data = content.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(upload_directory, name), "wb") as fp:
            fp.write(base64.decodebytes(data))


def save_dxf_to_csv(name, upload_directory):
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
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(upload_directory):
        path = os.path.join(upload_directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files
