from scipy.interpolate import interpn
import gpxpy
import gpxpy.gpx
from IPython.display import display
import sys
import webbrowser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
from math import radians,sqrt,cos,sin
import matplotlib.pyplot as plt
import folium
import random

import dash
from dash import dcc
from dash import html

a= 6378137 #wielka półoś
e2= 0.006694380022 #kwadrat pierwszego mimośrodu dla elipsoidy GRS80
def dozamiany(fi,lam,h):
    N = a / (sqrt(1 - (e2*(sin(radians(fi)))**2)))
    X = (N+h)*cos(radians(fi))*cos(radians(lam))
    Y = (N+h)*cos(radians(fi))*sin(radians(lam))
    Z = (N*(1-e2)+h)*sin(radians(fi))
    return X, Y, Z

gpx_file_name='okolice-janowa-podlaskiego.gpx'
with open (gpx_file_name,'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)
    route_info=[]
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({'latitude':point.latitude,'longitude':point.longitude,'elevation':point.elevation})


route_df=pd.DataFrame(route_info)
route_df['elevation_diff'] = route_df['elevation'].diff()




dist=[None]
for i in range(len(route_info)):
    if i==0:
        continue
    else:
        phi1= route_df.iloc[i-1]['latitude']
        lam1= route_df.iloc[i-1]['longitude']
        h1= route_df.iloc[i-1]['elevation']
        xyz1=dozamiany(phi1,lam1,h1)

        phi2= route_df.iloc[i]['latitude']
        lam2= route_df.iloc[i]['longitude']
        h2= route_df.iloc[i]['elevation']
        xyz2=dozamiany(phi2,lam2,h2)

        d= np.linalg.norm(np.array(xyz2)-np.array(xyz1))
        dist.append(d)


route_df['distance']= dist
route_df['cumul_dist']= route_df['distance'].cumsum()
route_df['cum_elevation'] = route_df['elevation_diff'].cumsum()
print(route_df)

#Wysokości elipsoidalne
fig = px.line(x=route_df['cumul_dist'],y=route_df['elevation'], title= 'Wysokości elipsoidalne')
fig.update_traces(mode="markers+lines", hovertemplate=None, marker_color='#DB7093', line_color='#FF69B4')
fig.update_layout(title_x=0.5, titlefont=dict(family='Courier New, monospace', size=18, color='#FF69B4'))
fig.update_layout(xaxis_title='Suma odległości od początku trasy', yaxis_title='Wysokości elipsoidalne', plot_bgcolor='#FFF5EE')
# fig.show()

#Trasa
import folium
route_map = folium.Map(
    #location=[53.909886,22.904866],
    #location=[52.156507,23.296589],
    location=[51.83529000000001,0.33044],
    zoom_start=11.5,
    tiles='CartoDBDark_Matter',
    width=1024,
    height=700
)

coordinates = [tuple(x) for x in route_df[['latitude', 'longitude']].to_numpy()]
folium.PolyLine(coordinates, weight=6, color='pink').add_to(route_map)

#display(route_map)

#Gradacja
gradients = [np.nan]

for ind, row in route_df.iterrows():
    if ind == 0 :
        continue
    try:
        grade = (row['elevation_diff'] / row['distance']) * 100
        gradients.append(np.round(grade, 1))
    except:
        continue



route_df['gradient'] = gradients
route_df['gradient'] = route_df['gradient'].interpolate().fillna(0)

# plt.title('Pochyłość na trasie', size=20)
# plt.xlabel('Punkt trasy', size=14)
# plt.ylabel('Pochyłość (%)', size=14)
# plt.plot(np.arange(len(route_df)), route_df['gradient'], lw=2, color='#101010')


fig1 = px.line(x=np.arange(len(route_df)),y=route_df['gradient'], title= 'Nachylenie terenu na trasie')
fig1.update_traces(mode="markers+lines", hovertemplate=None, marker_color='#DB7093', line_color='#FF69B4')
fig1.update_layout(title_x=0.5, titlefont=dict(family='Courier New, monospace', size=18, color='#FF69B4'))
fig1.update_layout(xaxis_title='Numer punktu trasy', yaxis_title='Nachylenie w %', plot_bgcolor='#FFF5EE')
# fig1.show()

#plt.show()

fig2 = px.line_mapbox(
    route_df,
    lat="latitude",
    lon="longitude",

    color_discrete_sequence=["fuchsia"],
    zoom=11,

)
fig2.update_layout(mapbox_style="open-street-map")
fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig2.show()

# wczytanie modelu geoidy

model= np.genfromtxt('Model_quasi-geoidy-PL-geoid2021-PL-EVRF2007-NH.txt',skip_header=True)



x = model[:, 0]
y = model[:, 1]
z = model[:, 2]
x_grid = x.reshape((len(np.unique(x)), -1))
y_grid = y.reshape((len(np.unique(x)), -1))
z_grid = z.reshape((len(np.unique(x)), -1))

x_range = x_grid[:, 0]
y_range = y_grid[0, :]

# interpolacja każdego punktu trasy, aby otrzymać zeta w danym punkcie trasy
zeta_all = []
for i in range(len(route_df)):
    fi = route_df.iloc[i]['latitude']
    lam = route_df.iloc[i]['longitude']
    zeta = interpn((x_range, y_range), z_grid, (fi, lam))
    zeta_all.append(float(zeta))

# nowa kolumna z wysokościami normalmnymi
route_df['normal'] = route_df['elevation'] - np.array(zeta_all)

fig3 = px.line(x=route_df['cumul_dist'],y=route_df['normal'],title= 'Wysokości normalne')
fig3.update_traces(mode="markers+lines", hovertemplate=None, marker_color='#DB7093', line_color='#FF69B4')
fig3.update_layout(title_x=0.5, titlefont=dict(family='Courier New, monospace', size=18, color='#FF69B4'))
fig3.update_layout(xaxis_title='Suma odległości od początku trasy', yaxis_title='Wysokości normalne', plot_bgcolor='#FFF5EE')
#fig3.show()


# model= np.genfromtxt('C:/Users/01169793/Desktop/Model_quasi-geoidy-PL-geoid2021-PL-EVRF2007-NH.txt',skip_header=True)
#
# #wysokości normalne
#
# x= model[:,0]
# y=model[:,1]
# z=model[:,2]
#
#
# x_grid= x.reshape((len(np.unique(x))),-1)
# y_grid = y.reshape((len(np.unique(y))),-1)
# z_grid = z.reshape((len(np.unique(z))),-1)
#
#
# x_range= x_grid[:,0]
# y_range= y_grid[0,:]
# zeta_all = []
#
#
# for i in range(len(route_df)):
#     fi= route_df.iloc[i]['latitude']
#     lam= route_df.iloc[i]['longtitude']
#     zeta= interpn((x_range,y_range),z_grid,(fi,lam))
#     zeta_all.append(float(zeta))
#
# route_df['normal']= route_df['elevation'] - np.array(zeta_all)
# fig = px.line(x=route_df['cumul_dist'],y=route_df['normal'])
# fig.show()

app = dash.Dash()
app.layout = html.Div(children=[
    dcc.Interval(id='animation-interval', interval=1000, n_intervals=0),
    html.H1(id='animated-title',children='Paulina Czerska 319263- projekt 5', className='title', style={'color': '#FF69B4', 'font-family': 'Courier New, monospace', 'font-size': 30, 'font-weight': 15}),
    html.Hr(style={'border': '2px dashed #FF69B4', 'border-style': 'dotted'}),
    dcc.Graph(id='graph1', figure=fig),
    html.Hr(style = {'border': '2px dashed #FF69B4'}),
    dcc.Graph(id='graph2', figure=fig1),
    html.Hr(style = {'border': '2px dashed #FF69B4'}),
    html.Br(),
    html.Div(children='Mapa przebiegu trasy',style = {'color': '#FF69B4', 'font-family':'Courier New, monospace', 'font-size':18,'text-align': 'center','font-weight':15}),
    html.Br(),
    dcc.Graph(id='graph3', figure=fig2),
    html.Br(),
    dcc.Graph(id='graph4', figure=fig3),
], style={'background-color': '#FAEBD7'})

@app.callback(
    dash.dependencies.Output('animated-title', 'style'),
    [dash.dependencies.Input('animation-interval', 'n_intervals')]
)
def update_title_style(n_intervals):
    colors = ['#FF69B4', '#00BFFF', '#7FFF00', '#FFD700', '#FF1493']
    return {'color': random.choice(colors)}
if __name__ == '__main__':
    url = "http://localhost:8050/"
    webbrowser.open(url, new=2)
    app.run_server(debug=True)

