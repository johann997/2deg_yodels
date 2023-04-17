# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
import numpy as np
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State


def create_n_sliders(num_sliders, app_layout):
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

    potential_layout, app_inputs = create_n_sliders(num_sliders, potential_layout)
    app_layout = setup_layout + potential_layout

    return app_layout, app_inputs
