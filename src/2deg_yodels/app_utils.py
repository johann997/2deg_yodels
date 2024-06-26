import numpy as np
from dash import Dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output, State


def create_n_sliders(num_sliders, app_layout):
    """
    Create num_sliders number of sliders. Each with key val_{i}-potential-slider

    Args:
        num_sliders (int): Number of sliders
        app_layout (html.Div()): input app_layout

    Returns:
        html.Div(): sliders are saved to html.Div and appended to input app_layout
    """
    slider_layouts = []
    app_inputs = []

    for i in range(num_sliders):
        key = f"val_{i}"

        slider_layout = [
            html.P(f"{key}"),
            dcc.RangeSlider(
                id=f"{key}-potential-slider",
                min=0,
                max=1,
                step=0.01,
                marks={i: "{}".format(i) for i in np.linspace(0, 1, 5)},
                value=[1.0],
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ]

        slider_layouts = slider_layouts + slider_layout

        app_input = [Input(f"{key}-potential-slider", "value")]
        app_inputs = app_inputs + app_input

    slider_layouts = html.Div(
        slider_layouts,
        style={"width": "39%", "float": "right", "display": "inline-block"},
    )

    app_layout.append(slider_layouts)

    return app_layout, app_inputs


def create_app_layout(initial_fig, UPLOAD_DIRECTORY):
    """
    Create base app layout

    Args:
        initial_fig (fig): fig to show before any data has been run
        UPLOAD_DIRECTORY (str): file path to save files

    Returns:
         html.Div: how app should be constructed
         list(Input()): Inputs of app
    """
    num_sliders = 20
    fig_style = {"verticalAlign": "middle", "width": "70vh", "height": "70vh"}
    # Dash Layouts
    setup_layout = [
        html.Div(
            [
                html.H1("2DEG yodel"),
                html.H2("Upload .dxf"),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                ),
                html.H3("Set the Inputs"),
                dcc.Graph(id="initial-gate-graph", figure=initial_fig, style=fig_style),
                html.H5("Depth of 2DEG"),
                dcc.Input(
                    id="2deg-depth",
                    type="number",
                    placeholder="2deg_depth in um",
                    value=0.09,
                ),
                html.H5("Zoom to [xmin, xmax] and [ymin, ymax]"),
                dcc.Input(
                    id="min-x-potential",
                    type="number",
                    placeholder="min x range",
                    value=9.6,
                ),
                dcc.Input(
                    id="max-x-potential",
                    type="number",
                    placeholder="max x range",
                    value=10.7,
                ),
                dcc.Input(
                    id="min-y-potential",
                    type="number",
                    placeholder="min y range",
                    value=0.5,
                ),
                dcc.Input(
                    id="max-y-potential",
                    type="number",
                    placeholder="max y range",
                    value=1.5,
                ),
                html.H5("Number of points in x direction and y direction"),
                dcc.Input(
                    id="numpts-x-potential",
                    type="number",
                    placeholder="numpts x",
                    value=50,
                ),
                dcc.Input(
                    id="numpts-y-potential",
                    type="number",
                    placeholder="numpts y",
                    value=50,
                ),
                html.Button("Update Gates", id="update-gate"),
                dcc.Graph(
                    id="discretised-gate-graph", figure=initial_fig, style=fig_style
                ),
                html.Button("Update Potential", id="update-potential"),
                daq.BooleanSwitch(
                    id="plot-potential-3d-switch",
                    on=False,
                    label="Plot 3d",
                    labelPosition="top",
                    color="#9B51E0",
                ),
            ]
        ),
        html.Div(id="dummy1"),
        html.Div(id="dummy2"),
        html.H4("Potential change in 2DEG"),
    ]

    potential_layout = [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="potential-graph"),
                    ],
                    style={"width": "59%", "height": "59%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(id="potential-slider-container-div", children=[]),
                        html.Div(id="potential-slider-container-output-div"),
                    ],
                    style={"width": "100%", "display": "inline-block"},
                ),
            ],
            style={"display": "flex"},
        )
    ]

    potential_plot = [
        html.Div(
            [
                html.H3("Plotting Potential vs. Gate"),
                html.H5("X Coord, Y Coord to Plot"),
                dcc.Input(
                    id="x-coord-pot",
                    type="value",
                    placeholder="X coordinate",
                    value="0",
                ),
                dcc.Input(
                    id="y-coord-pot",
                    type="value",
                    placeholder="Y coordinate",
                    value="0",
                ),
                html.H5("Gate 1 [id, min, max]"),
                dcc.Input(
                    id="gate1-pot-id",
                    type="text",
                    placeholder="gate number",
                    value="0",
                ),
                dcc.Input(
                    id="gate1-pot-min",
                    type="text",
                    placeholder="gate value min",
                    value="-1",
                ),
                dcc.Input(
                    id="gate1-pot-max",
                    type="text",
                    placeholder="gate value max",
                    value="0",
                ),
                html.Button(
                    "Run Potential off (1d plot)", id="run-pot-plot-1d", n_clicks=0
                ),
                dcc.Graph(id="pot-vs-gate", figure=initial_fig, style=fig_style),
            ],
            style={"width": "59%", "height": "59%", "display": "inline-block"},
        )
    ]

    kwant_layout = [
        html.Div(
            [
                html.H2("Transport Simulation"),
                html.H4("Lattice constant (nm)"),
                dcc.Input(
                    id="lattice-constant-kwant-system",
                    type="number",
                    placeholder="lattice constant (nm)",
                    value=1,
                ),
                html.H4("Lead coordinates"),
                html.H5("Lead 1 [x, y]"),
                dcc.Input(
                    id="lead1-x",
                    type="number",
                    placeholder="x coordinate",
                    value=9.6,
                ),
                dcc.Input(
                    id="lead1-y",
                    type="number",
                    placeholder="y coordinate",
                    value=0.5,
                ),
                html.H5("Lead 2 [x, y]"),
                dcc.Input(
                    id="lead2-x",
                    type="number",
                    placeholder="x coordinate",
                    value=9.6,
                ),
                dcc.Input(
                    id="lead2-y",
                    type="number",
                    placeholder="y coordinate",
                    value=1.25,
                ),
                html.H4("Lead size"),
                dcc.Input(
                    id="lead-length",
                    type="number",
                    placeholder="length (um)",
                    value=0.1,
                ),
                html.Button("Plot System", id="update-kwant-system"),
                html.Div(
                    [html.Img(id="kwant-system", src="")], id="kwant-system-plot-div"
                ),
                html.Div(
                    [html.Img(id="kwant-band-structure", src="")],
                    id="kwant-band-structure-plot-div",
                ),
                html.H3("Plotting wave function and current"),
                html.H4("Excitation energy at which to solve the scattering problem."),
                dcc.Input(
                    id="energy-kwant-simulation",
                    type="number",
                    placeholder="min numpts",
                    value=0.1,
                ),
                html.Button(
                    "Run wavefunction & Current", id="run-kwant-system-wf", n_clicks=0
                ),
                html.Div(
                    [html.Img(id="kwant-wave-function", src="")],
                    id="kwant-wave-function-plot-div",
                ),
                html.Div(
                    [html.Img(id="kwant-current", src="")], id="kwant-current-plot-div"
                ),
                html.H3("Run charge stability diagram"),
                html.H4("Min number of points in simulation"),
                dcc.Input(
                    id="numpts-kwant-simulation",
                    type="number",
                    placeholder="min numpts",
                    value=25,
                ),
                html.H5("Gate 1 [id, min, max]"),
                dcc.Input(
                    id="gate1-id",
                    type="value",
                    placeholder="gate number",
                    value="0",
                ),
                dcc.Input(
                    id="gate1-min",
                    type="number",
                    placeholder="gate value min",
                    value=-1,
                ),
                dcc.Input(
                    id="gate1-max",
                    type="number",
                    placeholder="gate value max",
                    value=0,
                ),
                html.Button(
                    "Run Pinch off (1d plot)", id="run-kwant-system-1d", n_clicks=0
                ),
                html.H5("Gate 2 [id, min, max]"),
                dcc.Input(
                    id="gate2-id",
                    type="value",
                    placeholder="gate number",
                    value="0",
                ),
                dcc.Input(
                    id="gate2-min",
                    type="number",
                    placeholder="gate value min",
                    value=-1,
                ),
                dcc.Input(
                    id="gate2-max",
                    type="number",
                    placeholder="gate value max",
                    value=0,
                ),
                html.Button(
                    "Run Charge Stability (2d plot)",
                    id="run-kwant-system-2d",
                    n_clicks=0,
                ),
                dcc.Graph(id="kwant-simulation", figure=initial_fig, style=fig_style),
            ],
            style={"width": "59%", "height": "59%", "display": "inline-block"},
        )
    ]

    app_layout = setup_layout + potential_layout + potential_plot + kwant_layout

    return app_layout
