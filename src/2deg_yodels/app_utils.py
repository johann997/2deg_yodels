import numpy as np
from dash import Dash
from dash import dcc
from dash import html
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
            ]
        ),
        html.Div(id="dummy1"),
        html.Div(id="dummy2"),
    ]


    potential_layout = [
        html.Div(
            [
                html.H4("Potential chane in 2DEG"),
                dcc.Graph(id="potential-graph"),
            ],
            style={"width": "59%", "height": "59%", "display": "inline-block"},
        )
    ]


    kwant_layout = [
        html.Div(
            [
                            html.H2("Transport Simulation"),
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
                html.Button("Plot System", id="update-kwant-system"),
                html.Img(id = 'kwant-system"', src = ''),
                # dcc.Graph(
                #     id="kwant-system", figure=initial_fig, style=fig_style
                # ),
                html.H4("Run charge stability diagram"),
                html.H5("Gate 1 [id, min, max]"),
                dcc.Input(
                    id="gate1-id",
                    type="number",
                    placeholder="gate number",
                    value=0,
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
                html.Button("Run Pinch off (1d plot)", id="run-kwant-system-1d", n_clicks=0),
                html.H5("Gate 2 [id, min, max]"),
                dcc.Input(
                    id="gate2-id",
                    type="number",
                    placeholder="gate number",
                    value=0,
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
                html.Button("Run Charge Stability (2d plot)", id="run-kwant-system-2d", n_clicks=0),
                dcc.Graph(
                    id="kwant-simulation", figure=initial_fig, style=fig_style
                ),
            ],
            style={"width": "59%", "height": "59%", "display": "inline-block"},
        )
    ]

    potential_layout, app_inputs = create_n_sliders(num_sliders, potential_layout)
    app_layout = setup_layout + potential_layout + kwant_layout

    return app_layout, app_inputs
