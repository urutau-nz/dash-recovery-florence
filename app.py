import pathlib
import os

import pandas as pd
import numpy as np
from urllib.request import urlopen
import json
import geopandas as gpd
import datetime


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

amenities = ['supermarket','gas_station']
amenity_names = {'supermarket':'Supermarket','gas_station':'Service Station'}

# app initialize
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=external_stylesheets,
    url_base_pathname='/resilience-florence/',
)
server = app.server
app.config["suppress_callback_exceptions"] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.title = 'Hurricane recovery'

# mapbox token
mapbox_access_token = open(".mapbox_token").read()

# Load data
df_dist = pd.read_csv('./data/distance_to_nearest.csv',dtype={"geoid10": str})
df_dist['distance'] = df_dist['distance']/1000
df_dist['distance'] = df_dist['distance'].replace(np.inf, 999)

destinations = pd.read_csv('./data/destinations.csv')

df_recovery = pd.read_csv('./data/recovery.csv')

# days since land landfall
days = np.unique(df_recovery['day'])

# Assign color to legend
colors = ['#EA5138','#E4AE36','#1F386B','#507332']
colormap = {}
for ind, amenity in enumerate(amenities):
    colormap[amenity] = colors[ind]

pl_deep=[[0.0, 'rgb(253, 253, 204)'],
         [0.1, 'rgb(201, 235, 177)'],
         [0.2, 'rgb(145, 216, 163)'],
         [0.3, 'rgb(102, 194, 163)'],
         [0.4, 'rgb(81, 168, 162)'],
         [0.5, 'rgb(72, 141, 157)'],
         [0.6, 'rgb(64, 117, 152)'],
         [0.7, 'rgb(61, 90, 146)'],
         [0.8, 'rgb(65, 64, 123)'],
         [0.9, 'rgb(55, 44, 80)'],
         [1.0, 'rgb(39, 26, 44)']]


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.A([
                html.Img(src=app.get_asset_url("urutau-logo.png")),
            ], href='https://apps.urutau.co.nz'),
            html.H6("Community resilience"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)

def brush(trace, points, state):
    inds = np.array(points.point_inds)
    if inds.size:
        selected = np.zeros(len(trace.x))
        selected[inds] = 1
        trace.marker.color = selected # now the trace marker color is a list of 0 and 1;
                                      # we have 0 at the position of unselected
                                      # points and 1 in the position of selected points

def generate_ecdf_plot(amenity_select, dff_dist, x_range=None):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    amenity = amenity_select
    if x_range is None:
        x_range = [dff_dist.distance.min(), dff_dist.distance.max()]


    layout = dict(
        xaxis=dict(
            title="distance to nearest {} (km)".format(amenity_names[amenity]).upper(),
            range=(0,15),
            ),
        yaxis=dict(
            title="% of residents".upper(),
            range=(0,100),
            ),
        font=dict(size=13),
        dragmode="select",
        paper_bgcolor = 'rgba(255,255,255,1)',
		plot_bgcolor = 'rgba(0,0,0,0)',
        bargap=0.05,
        showlegend=False,
        margin={'t': 10},
        # height= 300

    )
    data = []
    # add the cdf for that amenity
    counts, bin_edges = np.histogram(dff_dist.distance, bins=100, density = True)#, weights=df.W.values)
    dx = bin_edges[1] - bin_edges[0]
    new_trace = go.Scattergl(
            x=bin_edges, y=np.cumsum(counts)*dx*100,
            opacity=1,
            line=dict(color=colormap[amenity],),
            text=amenity_names[amenity]*len(dff_dist.service),
            hovertemplate = "%{y:.2f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )

    data.append(new_trace)

    # histogram
    multiplier = 300 if amenity=='supermarket' else 150
    counts, bin_edges = np.histogram(dff_dist.distance, bins=25, density=True)#, weights=df.W.values)
    opacity = []
    for i in bin_edges:
        if i >= x_range[0] and i <= x_range[1]:
            opacity.append(0.6)
        else:
            opacity.append(0.1)
    new_trace = go.Bar(
            x=bin_edges, y=counts*multiplier,
            marker_opacity=opacity,
            marker_color=colormap[amenity],
            hoverinfo="skip", hovertemplate="",)
    data.append(new_trace)


    # add the cdf for that amenity
    dff_dist = df_dist[(df_dist.day==days[0]) & (df_dist.service==amenity)]
    counts, bin_edges = np.histogram(dff_dist.distance, bins=100, density = True)#, weights=df.W.values)
    dx = bin_edges[1] - bin_edges[0]
    new_trace = go.Scattergl(
            x=bin_edges, y=np.cumsum(counts)*dx*100,
            opacity=0.5,
            line=dict(color=colormap[amenity]),
            text=[amenity_names[amenity].lower()]*len(dff_dist),
            # hovertemplate = "%{y:.2f}% of residents live within %{x:.1f}km of a %{text} <br>" + "<extra></extra>",
            # hoverlabel = dict(font_size=20),
            hoverinfo="skip", hovertemplate="",
            )

    data.append(new_trace)


    return {"data": data, "layout": layout}



def recovery_plot(amenity_select, dff_recovery, day):
    """
    :param amenity_select: the amenity of interest.
    :return: Figure object
    """
    amenity = amenity_select
    if amenity == 'supermarket':
        ylimit = 15
    else:
        ylimit = 8

    layout = dict(
        xaxis=dict(
            title="days since hurricane landfall".upper(),
            zeroline=False,
            ),
        yaxis=dict(
            title="Distance (km)".format(amenity_names[amenity]).upper(),

            zeroline=False,
            range=(ylimit,0),
            # autorange='reversed',
            ),
        font=dict(size=13),
        paper_bgcolor = 'rgba(255,255,255,1)',
		plot_bgcolor = 'rgba(0,0,0,0)',
        showlegend=False,
        margin={'t': 10},
        # height= 300

    )

    data = []
    # add the average
    new_trace = go.Scattergl(
            x=dff_recovery.day, y=dff_recovery['average']/1000,
            opacity=1,
            line=dict(color=colormap[amenity],),
            text=[amenity_names[amenity].lower()]*len(dff_recovery),
            hovertemplate = "The average distance to the nearest %{text} was %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)
    # add the percentiles
    new_trace = go.Scattergl(
            x=dff_recovery.day, y=dff_recovery['p5']/1000,
            opacity=.50,
            line=dict(color=colormap[amenity],dash='dash'),
            text=dff_recovery.service,
            hovertemplate = "5th % = %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)
    # add the percentiles
    new_trace = go.Scattergl(
            x=dff_recovery.day, y=dff_recovery['p95']/1000,
            opacity=.50,
            line=dict(color=colormap[amenity],dash='dash'),
            text=dff_recovery.service,
            hovertemplate = "95th % = %{y:.1f}km<br>" + "<extra></extra>",
            hoverlabel = dict(font_size=20),
            )
    data.append(new_trace)

    # add date line
    new_trace = go.Scattergl(
            x=[day, day],
            y=[0,ylimit+2],
            opacity=.50,
            mode='lines',
            line=dict(color='black',dash='dash'),
            hoverinfo="skip", hovertemplate="",
            )
    data.append(new_trace)

    return {"data": data, "layout": layout}



def generate_map(amenity, dff_dist, dff_dest, x_range=None):
    """
    Generate map showing the distance to services and the locations of them

    :param amenity: the service of interest.
    :param dff_dest: the lat and lons of the service.
    :param x_range: distance range to highlight.
    :return: Plotly figure object.
    """
    # print(dff_dist['geoid10'].tolist())
    dff_dist = dff_dist.reset_index()


    layout = go.Layout(
        clickmode="none",
        dragmode="zoom",
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        # height= 561,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat = 34.245580, lon = -77.872072),
            pitch=0,
            zoom=10.5,
            style="basic", #"dark", #
        ),
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="bottom",
        ),
    )

    if x_range:
        # get the indices of the values within the specified range
        idx = dff_dist.index[dff_dist['distance'].between(x_range[0],x_range[1], inclusive=True)].tolist()
    else:
        idx = dff_dist.index.tolist()

    data = []
    # choropleth map showing the distance at the block level
    data.append(go.Choroplethmapbox(
        geojson = 'https://raw.githubusercontent.com/urutau-nz/dash-recovery-florence/master/data/block.geojson',
        locations = dff_dist['geoid10'].tolist(),
        z = dff_dist['distance'].tolist(),
        colorscale = pl_deep,
        colorbar = dict(thickness=20, ticklen=3), zmin=0, zmax=5,
        marker_line_width=0, marker_opacity=0.7,
        visible=True,
        hovertemplate="Distance: %{z:.2f}km<br>" +
                        "<extra></extra>",
        selectedpoints=idx,
    ))

    # scatterplot of the amenity locations
    point_color = [colormap[amenity] if i==True else 'black' for i in dff_dest['operational']]

    data.append(go.Scattermapbox(
        lat=dff_dest["lat"],
        lon=dff_dest["lon"],
        mode="markers",
        marker={"color": point_color, "size": 9},
        # marker={"color": dff_dest['operational'], "size": 9},
        name=amenity_names[amenity],
        hoverinfo="skip", hovertemplate="",
    ))

    return {"data": data, "layout": layout}


app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children=dcc.Markdown('''
                                    To make communities resilient, we need to think about what they require to
                                    function. It's not just the infrastructure, but the services, amenities, and
                                    opportunities that that enables: health care, groceries, education, employment,
                                    etc. In [Logan et. al (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1111/risa.13492) we propose a new way to think about making our
                                    communities resilient.  \n
                                    This is an example where we evaluate how access to supermarkets and service stations
                                    changed over the course of a hurricane. This is based on Hurricane Florence,
                                    which hit Wilmington, NC, in 2018.
                                    '''),
                                ),
                                build_graph_title("Select Amenity"),
                                dcc.Dropdown(
                                    id="amenity-select",
                                    options=[
                                        {"label": amenity_names[i].upper(), "value": i}
                                        for i in amenities
                                    ],
                                    value=amenities[0],
                                ),
                            ],
                        )
                    ],
                ),
                # html.Div(
                #     className="row",
                #     # id="top-row-graphs",
                #     children=[
                #         # Access map
                #         html.Div(
                #             children=[
                #
                #             ],
                #         ),
                #     ],
                # ),
                html.Div(
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Access map
                        html.Div(
                            id="map-container",
                            children=[
                                build_graph_title("Select the day since landfall"),
                                dcc.Slider(
                                    id="day-select",
                                    min=np.min(days),
                                    max=np.max(days),
                                    # step=2,
                                    marks={i: str(i) for i in range(np.min(days),np.max(days),1)},
                                    value=-2,
                                ),
                                build_graph_title("How has people's access to services changed?"),
                                dcc.Graph(
                                    id="map",
                                    figure={
                                        "layout": {
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                        }
                                    },
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":["lasso2d","select2d"],
                                    },
                                ),
                                # build_graph_title("Select the day since landfall"),
                                # dcc.Slider(
                                #     min=0,
                                #     max=9,
                                #     marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
                                #     value=5,
                                # ),
                            ],
                        ),
                        # ECDF
                        html.Div(
                            id="ecdf-container",
                            className="six columns",
                            children=[
                                build_graph_title("Select a distance range to identify those areas"),
                                dcc.Graph(id="ecdf",
                                    # figure={
                                    #     "layout": {
                                    #         # 'clickmode': 'event+select',
                                    #         "paper_bgcolor": "#192444",
                                    #         "plot_bgcolor": "#192444",
                                    #         'mode': 'markers+lines',
                                    #         'margin': {
                                    #             'l': 0,
                                    #             'r': 0,
                                    #             'b': 0,
                                    #             't': 0,
                                    #             'pad': 0
                                    #           }
                                    #     }
                                    # },
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian'],
                                    },
                                ),
                                build_graph_title("Resilience curve"),
                                dcc.Graph(id="recovery",
                                    # figure={
                                    #     "layout": {
                                    #         'height': '32vh',
                                    #         "paper_bgcolor": "#192444",
                                    #         "plot_bgcolor": "#192444",
                                    #         # 'mode': 'markers+lines',
                                    #         'shapes':{
                                    #             'type':'line',
                                    #             'y0': 20, 'y1': 0,
                                    #             # 'xref': 'x0',
                                    #             'x0': 5, 'x1': 5,
                                    #         }
                                    #     }
                                    # },
                                    config={"scrollZoom": True, "displayModeBar": True,
                                            "modeBarButtonsToRemove":['toggleSpikelines','hoverCompareCartesian'],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="footer-row",
            children=[
                html.P(
                    id="footer-text",
                    children=dcc.Markdown('''
                        Thank you to the developers of [Dash and Plotly]
                        (https://plotly.com/dash/), whose work made this app possible.
                        '''
                    ),
                )
            ]
        )
    ]
)


# Update access map
@app.callback(
    Output("map", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
        Input("ecdf", "selectedData"),
    ],
)
def update_map(
    amenity_select, day, ecdf_selectedData
):
    x_range = None
    day = int(day)
    # subset the desination df
    dff_dest = destinations[(destinations.dest_type==amenity_select) & (destinations['day']==day)]
    dff_dist = df_dist[(df_dist['service']==amenity_select) & (df_dist['day']==day)]
    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2

    return generate_map(amenity_select, dff_dist, dff_dest, x_range=x_range)


# Update ecdf
@app.callback(
    Output("ecdf", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
        Input("ecdf", "selectedData"),
    ],
)
def update_ecdf(
    amenity_select, day, ecdf_selectedData
    ):
    x_range = None
    # day = int(day)

    # subset data
    dff_dist = df_dist[(df_dist['service']==amenity_select) & (df_dist['day']==day)]

    # Find which one has been triggered
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]

    if prop_id == 'ecdf' and prop_type == "selectedData":
        if ecdf_selectedData:
            if 'range' in ecdf_selectedData:
                x_range = ecdf_selectedData['range']['x']
            else:
                x_range = [ecdf_selectedData['points'][0]['x']]*2

    return generate_ecdf_plot(amenity_select, dff_dist, x_range)

# Update ecdf
@app.callback(
    Output("recovery", "figure"),
    [
        Input("amenity-select", "value"),
        Input("day-select", "value"),
    ],
)
def update_recovery(
    amenity_select, day
    ):
    x_range = None
    day = int(day)

    # subset data
    dff_recovery = df_recovery[(df_recovery['service']==amenity_select)]

    return recovery_plot(amenity_select, dff_recovery, day)



# Running the server
if __name__ == "__main__":
    # app.run_server(debug=True, port=8050)
   app.run_server(port=9005)
