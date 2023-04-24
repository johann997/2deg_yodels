print("Running app.py")
import os
import numpy as np
from dash import Dash
from dash import dcc
from dash import html
from dash import ctx
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

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
    fig_to_uri
)

from kwant_utils import (
    make_kwant_system,
    plot_kwant_system,
    plot_kwant_potential,
    plot_kwant_info,
    get_kwant_transmission
)

cwd = os.getcwd()
base = cwd.split('2deg_yodels')[0]
path_to_save = f'{base}2deg_yodels/src/2deg_yodels'

UPLOAD_DIRECTORY = f"{path_to_save}/temp_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

PROCESSED_DIRECTORY = f"{path_to_save}/temp_processed_data"
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)

app_layout, app_inputs = create_app_layout(default_fig(), UPLOAD_DIRECTORY)

app = Dash(__name__)

app.layout = html.Div(app_layout)


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
    Output("dummy2", "children"),
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
        return None


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
    app_inputs,
)
def update_potential(
    depth_2deg, minx, maxx, miny, maxy, nx, ny, filename, *slider_vals
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )
        z_data = 0
        index = -1
        for key, val in discretised_gates.items():
            if "val_" in key:
                index += 1
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
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
        )

        return potential_fig
    else:
        return default_fig()


#################################
##### creating kwant system #####
#################################
@app.callback(
    Output("kwant-system", component_property='src'),
    Input("update-kwant-system", "n_clicks"),
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    app_inputs,
)
def update_kwant_system(
    update_kwant_system, lead1x, lead1y, lead2x, lead2y, depth_2deg, minx, maxx, miny, maxy, nx, ny, filename, *slider_vals
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )
        z_data = 0
        index = -1
        for key, val in discretised_gates.items():
            if "val_" in key:
                index += 1
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]
        numpts = 100

        qpc = make_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts)

        fig = plot_kwant_system(qpc)
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
    State("lead1-x", "value"),
    State("lead1-y", "value"),
    State("lead2-x", "value"),
    State("lead2-y", "value"),
    State("gate1-id", "value"),
    State("gate1-min", "value"),
    State("gate1-max", "value"),
    State("gate2-id", "value"),
    State("gate2-min", "value"),
    State("gate2-max", "value"),
    State("2deg-depth", "value"),
    State("min-x-potential", "value"),
    State("max-x-potential", "value"),
    State("min-y-potential", "value"),
    State("max-y-potential", "value"),
    State("numpts-x-potential", "value"),
    State("numpts-y-potential", "value"),
    State("upload-data", "filename"),
    app_inputs,
)
def update_kwant_system(
    update_kwant_system, lead1x, lead1y, lead2x, lead2y, gate1id, gate1min, gate1max, gate2id, gate2min, gate2max,  depth_2deg, minx, maxx, miny, maxy, nx, ny, filename, *slider_vals
):
    if filename is not None:
        discretised_gates = get_discretised_gates_from_csv(
            nx, ny, csv_name=f"{PROCESSED_DIRECTORY}/geometric_potential.csv"
        )
        z_data = 0
        index = -1
        for key, val in discretised_gates.items():
            if "val_" in key:
                index += 1
                discretised_gates[key]["gate_val"] = slider_vals[index][0]
                z_data = z_data + discretised_gates[key]["potential"]

        lead1_coords = np.array([lead1x, lead1y])
        lead2_coords = np.array([lead2x, lead2y])
        lead_coords = [lead1_coords, lead2_coords]
        numpts_system = 100

        # looping for multiple gate vals
        gate1_name = f"val_{int(gate1id)}"
        gate2_name = f"val_{int(gate2id)}"
        numpts_simulation = 50
        transmission_array = np.zeros((numpts_simulation, numpts_simulation))
        voltage1_array = np.linspace(gate1min, gate1max, numpts_simulation)
        voltage2_array = np.linspace(gate2min, gate2max, numpts_simulation)

        for v1_index, voltage_1 in enumerate(voltage1_array):
            discretised_gates[gate1_name]["gate_val"] = voltage_1

            if "run-kwant-system-1d" == ctx.triggered_id:
                qpc = make_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts_system)
                transmission = get_kwant_transmission(qpc, energy=0, lead_out=1, lead_in=0)
                transmission_array[0, v1_index] = transmission  

            elif "run-kwant-system-2d" == ctx.triggered_id:
                for v2_index, voltage_2 in enumerate(voltage2_array):
                    discretised_gates[gate2_name]["gate_val"] = voltage_2
                    qpc = make_kwant_system(discretised_gates, lead_coords, minx, maxx, miny, maxy, numpts_system)

                    transmission = get_kwant_transmission(qpc, energy=0, lead_out=1, lead_in=0)
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

        elif "run-kwant-system-2d" == ctx.triggered_id:
            fig.add_trace(
                go.Heatmap(
                    z=transmission_array,
                    x=voltage1_array,
                    y=voltage2_array,
                    opacity=1,
                )
            )

        return fig
    else:
        return default_fig()
    

# run the app
if __name__ == "__main__":
    app.run_server(port=4444, debug=False)
