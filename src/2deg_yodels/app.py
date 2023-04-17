print("Running app.py")
import os
import numpy as np
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

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
)

UPLOAD_DIRECTORY = f"{os.getcwd()}/temp_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

PROCESSED_DIRECTORY = f"{os.getcwd()}/temp_processed_data"
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)

app_layout, app_inputs = create_app_layout(default_fig(), UPLOAD_DIRECTORY)

app = Dash(__name__)

app.layout = html.Div(app_layout)


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
##### updating discretisation #####
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


# run the app
if __name__ == "__main__":
    app.run_server(port=4444, debug=False)
