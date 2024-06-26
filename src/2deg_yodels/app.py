print("Running app.py")
import os
import numpy as np
from dash import Dash
from dash import dcc
from dash import html
from dash import ctx
from dash import ALL
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

from app_utils import create_app_layout, create_n_sliders

from utils import (
    default_fig,
    plot_discretised_gates,
    get_potential_from_gate,
    save_geometric_potential_to_csv,
    get_discretised_gates_from_csv,
    get_discretised_gates,
    get_plot_info,
    save_file,
    save_dxf_to_csv,
    read_csv_to_polyline,
    plot_polyline,
    fig_to_uri,
    get_xyz_from_discretised_gates,
)

from kwant_utils import (
    make_kwant_system,
    plot_kwant_system,
    plot_kwant_potential,
    plot_kwant_info,
    get_kwant_transmission,
    plot_kwant_band_structure,
    plot_kwant_density,
    get_interpolated_potential,
)

cwd = os.getcwd()
base = cwd.split("2deg_yodels")[0]
path_to_save = f"{base}2deg_yodels/src/2deg_yodels"

UPLOAD_DIRECTORY = f"{path_to_save}/temp_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

PROCESSED_DIRECTORY = f"{path_to_save}/temp_processed_data"
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)

app_layout = create_app_layout(default_fig(), UPLOAD_DIRECTORY)

# external_stylesheets=[dbc.themes.BOOTSTRAP]

app = Dash(__name__)

app.title = "2DEG_YODELS"
app.layout = html.Div(
    app_layout,
)


#######################################################
##### save .dxf to UPLOAD_DIRECTORY and plot data #####
#######################################################
@app.callback(
    Output("initial-gate-graph", "figure"),
    Input("upload-data", "filename"),
    Input("upload-data", "contents"),
)
def update_output(filename, uploaded_file_content):
    """Save uploaded files and regenerate the file list."""
    if filename is not None:
        save_file(filename, uploaded_file_content, UPLOAD_DIRECTORY)
        save_dxf_to_csv(filename, UPLOAD_DIRECTORY)
        return plot_polyline(filename, UPLOAD_DIRECTORY)
    else:
        return default_fig()


###################################
##### updating discretisation #####
###################################
@app.callback(
    Output("discretised-gate-graph", "figure"),
    Input("update-gate", "n_clicks"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
)
def update_discretised_gates(
    update_gate, depth_2deg, minx, maxx, miny, maxy, nx, ny, filename
):
    if filename is not None:
        polyline_gates = read_csv_to_polyline(filename, UPLOAD_DIRECTORY)

        plot_info = get_plot_info(depth_2deg, minx, maxx, miny, maxy, nx, ny)
        discretised_gates = get_discretised_gates(plot_info, polyline_gates)
        discretised_fig = plot_discretised_gates(
            discretised_gates,
            plot_info,
            plot=False,
        )

        return discretised_fig
    else:
        return default_fig()


###################################
#####   calculate potential   #####
###################################
@app.callback(
    Output("potential-slider-container-div", "children"),
    Input("update-potential", "n_clicks"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
)
def update_potential_csv(
    update_potential_csv, depth_2deg, minx, maxx, miny, maxy, nx, ny, filename
):
    if filename is not None:
        polyline_gates = read_csv_to_polyline(filename, UPLOAD_DIRECTORY)

        plot_info = get_plot_info(depth_2deg, minx, maxx, miny, maxy, nx, ny)
        discretised_gates = get_discretised_gates(plot_info, polyline_gates)
        material_info = {"2deg_depth": depth_2deg}
        discretised_gates_new = get_potential_from_gate(
            discretised_gates, material_info
        )
        save_geometric_potential_to_csv(
            discretised_gates_new,
            csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv",
        )

        slider_layouts = []
        for key, val in discretised_gates.items():
            if "val_" in key:
                slider_layout = [
                    html.P(f"{key}"),
                    dcc.RangeSlider(
                        id={
                            "type": "potential-slider",
                            "index": f"{key.split('_')[-1]}",
                        },
                        # id=f"{key}-potential-slider",
                        min=-1,
                        max=0,
                        step=0.01,
                        marks={i: "{}".format(i) for i in np.linspace(-1, 0, 5)},
                        value=[-1.0],
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]

                slider_layouts = slider_layouts + slider_layout

        slider_layouts = html.Div(
            slider_layouts,
            style={"width": "49%", "float": "right", "display": "inline-block"},
        )

        return [slider_layouts]


##############################
##### updating potential #####
##############################
@app.callback(
    Output("potential-graph", "figure"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State("potential-graph", "relayoutData"),
    Input({"type": "potential-slider", "index": ALL}, "value"),
    Input("plot-potential-3d-switch", "on"),
)
def update_potential(
    depth_2deg,
    minx,
    maxx,
    miny,
    maxy,
    nx,
    ny,
    filename,
    dash_figure,
    slider_vals,
    plot_potential_3d,
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )
        z_data = 0
        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = -slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        color_range = [np.min(z_data), np.max(z_data)]
        plot_info = get_plot_info(depth_2deg, minx, maxx, miny, maxy, nx, ny)

        potential_fig = plot_discretised_gates(
            discretised_gates,
            plot_info,
            plot_type="potential",
            plot=False,
            colorscale="Plotly3",
            color_range=color_range,
            plot_3d=plot_potential_3d,
        )

        potential_fig.update_layout(height=800, width=800)

        for key, val in dash_figure.items():
            if "scene.camera" in key:
                potential_fig.update_layout(scene_camera=val)

        return potential_fig
    else:
        return default_fig()



######################################################
##### running potential plotting vs gate system  #####
######################################################
@app.callback(
    Output("pot-vs-gate", "figure"),
    Input("run-pot-plot-1d", "n_clicks"),
    State("x-coord-pot", "value"),
    State("y-coord-pot", "value"),
    State("gate1-pot-id", "value"),
    State("gate1-pot-min", "value"),
    State("gate1-pot-max", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State({"type": "potential-slider", "index": ALL}, "value"),
)
def plot_potential_vs_gate(
    update_pot_vs_gate,
    x_val,
    y_val,
    gates_id,
    gates_min,
    gates_max,
    nx,
    ny,
    filename,
    slider_vals,
):
    
    # print(gates_id, gates_min, gates_max)
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )

        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = slider_vals[index][0]      

        gate1_array = np.array(gates_id.split(","))
        voltage_min_array = np.array(gates_min.split(",")).astype(float)
        voltage_max_array = np.array(gates_max.split(",")).astype(float)
        numpts = 100

        potential_array = np.linspace(0, 1, numpts)

        for pot_index in range(numpts):  # hard code 100 points for now
            for gate_index, gate1_str in enumerate(gate1_array):
                voltage = pot_index*((voltage_max_array[gate_index] - voltage_min_array[gate_index])/numpts) + voltage_min_array[gate_index]
                discretised_gates[f"val_{int(gate1_str)}"]["gate_val"] = -voltage
            x, y, potential = get_xyz_from_discretised_gates(discretised_gates, data_type="potential")
            x_coord = np.argmin(np.abs(x - float(x_val)))
            y_coord = np.argmin(np.abs(y - float(y_val)))

            potential_array[pot_index] = potential[y_coord, x_coord]

        # setting up figure for plotting
        fig = default_fig()

        x_axis_voltage = np.linspace(voltage_min_array[0], voltage_max_array[0], numpts)
        
        if "run-pot-plot-1d" == ctx.triggered_id:
            fig.add_trace(
                go.Scatter(
                    x=x_axis_voltage,
                    y=potential_array,
                )
            )
            fig.update_layout(
                xaxis_title=f"Gate Voltage {gate1_array} (V)", yaxis_title="Potential (V)"
            )

        return fig
    else:
        return default_fig()





#################################
##### creating kwant system #####
#################################
@app.callback(
    Output("kwant-system", component_property="src"),
    Input("update-kwant-system", "n_clicks"),
    State("lattice-constant-kwant-system", "value"),
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("lead-length", "value"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State({"type": "potential-slider", "index": ALL}, "value"),
)
def update_kwant_system(
    update_kwant_system,
    lattice_constant,
    lead1x,
    lead1y,
    lead2x,
    lead2y,
    lead_length,
    depth_2deg,
    minx,
    maxx,
    miny,
    maxy,
    nx,
    ny,
    filename,
    slider_vals,
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )

        z_data = 0
        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]

        qpc = make_kwant_system(
            discretised_gates,
            lead_coords,
            minx,
            maxx,
            miny,
            maxy,
            0,
            a=lattice_constant,
            lead_length=lead_length,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig = plot_kwant_system(qpc, ax=ax)
        fig = ax.get_figure()
        out_fig = fig_to_uri(fig)

        return out_fig
    else:
        return ""


#################################
##### plotting kwant bands  #####
#################################
@app.callback(
    Output("kwant-band-structure", component_property="src"),
    Input("update-kwant-system", "n_clicks"),
    State("lattice-constant-kwant-system", "value"),
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("lead-length", "value"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State({"type": "potential-slider", "index": ALL}, "value"),
)
def update_kwant_band_structure(
    update_kwant_band_structure,
    lattice_constant,
    lead1x,
    lead1y,
    lead2x,
    lead2y,
    lead_length,
    depth_2deg,
    minx,
    maxx,
    miny,
    maxy,
    nx,
    ny,
    filename,
    slider_vals,
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )

        z_data = 0
        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]

        qpc = make_kwant_system(
            discretised_gates,
            lead_coords,
            minx,
            maxx,
            miny,
            maxy,
            0,
            a=lattice_constant,
            lead_length=lead_length,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        fig = plot_kwant_band_structure(qpc, ax=ax)
        fig = ax.get_figure()
        out_fig = fig_to_uri(fig)

        return out_fig
    else:
        return ""


########################################
##### plotting kwant wavefunction  #####
########################################
@app.callback(
    Output("kwant-wave-function", component_property="src"),
    Input("run-kwant-system-wf", "n_clicks"),
    State("lattice-constant-kwant-system", "value"),
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("lead-length", "value"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State({"type": "potential-slider", "index": ALL}, "value"),
    State("energy-kwant-simulation", "value"),
)
def update_kwant_wavefunction(
    update_kwant_band_structure,
    lattice_constant,
    lead1x,
    lead1y,
    lead2x,
    lead2y,
    lead_length,
    depth_2deg,
    minx,
    maxx,
    miny,
    maxy,
    nx,
    ny,
    filename,
    slider_vals,
    energy_simulation,
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )

        z_data = 0
        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]

        qpc = make_kwant_system(
            discretised_gates,
            lead_coords,
            minx,
            maxx,
            miny,
            maxy,
            0,
            a=lattice_constant,
            lead_length=lead_length,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        fig = plot_kwant_band_structure(qpc, ax=ax)
        fig = plot_kwant_density(qpc, energy_simulation, lead_num=0, ax=ax)
        fig = ax.get_figure()
        out_fig = fig_to_uri(fig)

        return out_fig
    else:
        return ""


#################################
##### running kwant system  #####
#################################
@app.callback(
    Output("kwant-simulation", "figure"),
    Input("run-kwant-system-1d", "n_clicks"),
    Input("run-kwant-system-2d", "n_clicks"),
    State("lattice-constant-kwant-system", "value"),
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("lead-length", "value"),
    State("gate1-id", "value"),
    State("gate1-min", "value"),
    State("gate1-max", "value"),
    State("gate2-id", "value"),
    State("gate2-min", "value"),
    State("gate2-max", "value"),
    State("2deg-depth", "value"),
    State("numpts-kwant-simulation", "value"),
    State("energy-kwant-simulation", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    State({"type": "potential-slider", "index": ALL}, "value"),
)
def running_kwant_system(
    update_kwant_system1d,
    update_kwant_system2d,
    lattice_constant,
    lead1x,
    lead1y,
    lead2x,
    lead2y,
    lead_length,
    gate1id,
    gate1min,
    gate1max,
    gate2id,
    gate2min,
    gate2max,
    depth_2deg,
    numpts_simulation,
    energy_simulation,
    minx,
    maxx,
    miny,
    maxy,
    nx,
    ny,
    filename,
    slider_vals,
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )

        z_data = 0
        for index, (key, val) in enumerate(discretised_gates.items()):
            if "val_" in key:
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]

        transmission_array = np.zeros((numpts_simulation, numpts_simulation))
        voltage1_array = np.linspace(gate1min, gate1max, numpts_simulation)
        voltage2_array = np.linspace(gate2min, gate2max, numpts_simulation)

        gate1_array = gate1id.split(",")

        for v1_index, voltage_1 in enumerate(voltage1_array):
            for gate1_str in gate1_array:
                discretised_gates[f"val_{int(gate1_str)}"]["gate_val"] = voltage_1

            if "run-kwant-system-1d" == ctx.triggered_id:
                qpc = make_kwant_system(
                    discretised_gates,
                    lead_coords,
                    minx,
                    maxx,
                    miny,
                    maxy,
                    0,
                    a=lattice_constant,
                    lead_length=lead_length,
                )
                transmission = get_kwant_transmission(
                    qpc, energy=energy_simulation, lead_out=1, lead_in=0
                )
                transmission_array[0, v1_index] = transmission

            elif "run-kwant-system-2d" == ctx.triggered_id:
                gate2_array = gate2id.split(",")
                for v2_index, voltage_2 in enumerate(voltage2_array):
                    for gate2_str in gate2_array:
                        discretised_gates[f"val_{int(gate2_str)}"][
                            "gate_val"
                        ] = voltage_2
                    qpc = make_kwant_system(
                        discretised_gates,
                        lead_coords,
                        minx,
                        maxx,
                        miny,
                        maxy,
                        0,
                        a=lattice_constant,
                        lead_length=lead_length,
                    )

                    transmission = get_kwant_transmission(
                        qpc, energy=energy_simulation, lead_out=1, lead_in=0
                    )
                    transmission_array[v2_index, v1_index] = transmission

        # setting up figure for plotting
        fig = default_fig()

        if "run-kwant-system-1d" == ctx.triggered_id:
            fig.add_trace(
                go.Scatter(
                    x=voltage1_array,
                    y=transmission_array[0],
                )
            )
            fig.update_layout(
                xaxis_title="Gate Voltage (V)", yaxis_title="Conductance (e^2/h)"
            )

        elif "run-kwant-system-2d" == ctx.triggered_id:
            fig.add_trace(
                go.Heatmap(
                    z=transmission_array,
                    x=voltage1_array,
                    y=voltage2_array,
                    opacity=1,
                )
            )
            fig.update_layout(
                xaxis_title="Gate 1 Voltage (V)",
                yaxis_title="Gate 2 Voltage (V)",
            )

        return fig
    else:
        return default_fig()


# run the app
if __name__ == "__main__":
    app.run_server(port=4444, debug=False)
