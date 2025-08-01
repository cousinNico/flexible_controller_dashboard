import pandas as pd
import plotly.graph_objects as go

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

df = pd.read_csv("sims/simout.csv")

signal_names = df.columns[1:]  # Skip 'Time' column

pos_signals = [col for col in df.columns if (col.startswith('pos_') and not col.endswith('des'))]
posdes_signals = [col for col in df.columns if (col.startswith('pos_') and col.endswith('des'))]
att_signals = [col for col in df.columns if (col.startswith('att_') and not col.endswith('des'))]
attdes_signals = [col for col in df.columns if (col.startswith('att_') and col.endswith('des'))]
vel_signals = [col for col in df.columns if (col.startswith('vel_') and not col.endswith('des'))]
veldes_signals = [col for col in df.columns if (col.startswith('vel_') and col.endswith('des'))]
omega_signals = [col for col in df.columns if (col.startswith('omega_')and not col.endswith('des'))]
omegades_signals = [col for col in df.columns if (col.startswith('omega_') and col.endswith('des'))]
xi_signals = [col for col in df.columns if col.startswith('xi_')]
input_signals = [col for col in df.columns if col.startswith('u_')]
thetahat_signals = [col for col in df.columns if col.startswith('ThetaHat_')]

def plot_signals(fig, value, desvalue, dff, colors):
    for i, sig in enumerate(value):
            fig.add_trace(go.Scatter(
                x=dff['Time'], y=dff[sig],
                mode='lines',
                name=sig,
                line=dict(color=colors[i % len(colors)], dash='solid')
            ))
    # Plot posdes_signals (dashed)
    for i, sig in enumerate(desvalue):
        fig.add_trace(go.Scatter(
            x=dff['Time'], y=dff[sig],
            mode='lines',
            name=sig,
            line=dict(color=colors[i % len(colors)], dash='dash')
        ))
    return fig



app = Dash()
app.layout = html.Div([
    html.H1(children='Results Adaptive Control Flexible Spacecraft', style={'textAlign':'center'}),
    
    html.Div([
        html.H2('State Display', style={'margin-right': '20px'}),
        dcc.RadioItems(['Position Hub','Attitude Hub','Velocity Hub','Angular Velocity Hub','TFC Coordinates Flex','Inputs','Estimates Unknowns'], 'Attitude Hub', id='radio-selection'),
        html.H2('Select Controller', style={'margin-right': '20px'}),
        dcc.RadioItems(
        id='dropdown-selection',
        options=[
            {'label': 'LQR Rigid', 'value': 0},
            {'label': 'LQR Flexible', 'value': 1},
            {'label': 'Lyapunov Nonlinear Control', 'value': 2},
            {'label': 'Lyapunov Adaptive Control', 'value': 3},
            {'label': 'Integral Concurrent Learning Adaptive Control', 'value': 4}],
        value=0,
    ),
    ],style={'margin-top': '20px','display': 'flex', 'width': '100%'}),
    
    html.Div([
        dcc.Graph(id='graph-content', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='custom-graph-content', style={'width': '50%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        dcc.Slider(
            df['Time'].min(),
            df['Time'].max(),
            step=1,
            value=df['Time'].min(),
            id='time-slider'
        )])
])

#TODO: change the slider to be about the tuning for ICL rather than time

@callback(
    Output('graph-content', 'figure'),
    Input('radio-selection', 'value'),
    Input('dropdown-selection', 'value'),
    Input('time-slider', 'value'),
)

def update_graph(value, data_value, time_value):
    """Update the graph based on the selected value."""
    df = pd.read_csv("sims/tfc_single"+str(data_value)+"_simout.csv")
    dff = df[df['Time'] <= time_value]
    fig = go.Figure()
    fig.update_layout(
        title=f"Simulation Data for {value}",
        xaxis_title="Time",
        yaxis_title="Signal Value",
        template="plotly_dark"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')  

    colors = px.colors.qualitative.Plotly
    if value == 'Position Hub':
        return plot_signals(fig, pos_signals, posdes_signals, dff, colors)
    elif value == 'Attitude Hub':
        return plot_signals(fig, att_signals, attdes_signals, dff, colors)
    elif value == 'Velocity Hub':
        return plot_signals(fig, vel_signals, veldes_signals, dff, colors)
    elif value == 'Angular Velocity Hub':
        return plot_signals(fig, omega_signals, omegades_signals, dff, colors)
    elif value == 'TFC Coordinates Flex':
        return plot_signals(fig, xi_signals, [], dff, colors)
    elif value == 'Inputs':
        return plot_signals(fig, input_signals, [], dff, colors)
    elif value == 'Estimates Unknowns':
        return plot_signals(fig, thetahat_signals, [], dff, colors)
    else:
        return fig.add_traces(px.line(dff, x='Time', y=signal_names, title=value).data)

@callback(
    Output('custom-graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)

def get_custom_figure(data_value):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def create_arrow(start, vec, color='cyan', name='', arrow_scale=0.1,dash = 'solid'):
        """Create a quiver-style arrow using lines for shaft and head (dark mode)."""
        shaft = go.Scatter3d(
            x=[start[0], start[0] + vec[0]],
            y=[start[1], start[1] + vec[1]],
            z=[start[2], start[2] + vec[2]],
            mode='lines',
            line=dict(color=color, width=3),
            name=name,
            showlegend=True
        )
        tip = start + vec
        head = go.Scatter3d(
            x=[tip[0], tip[0] - arrow_scale * vec[0]],
            y=[tip[1], tip[1] - arrow_scale * vec[1]],
            z=[tip[2], tip[2] - arrow_scale * vec[2]],
            mode='lines',
            line=dict(color=color, width=5, dash=dash),
            showlegend=False
        )
        return [shaft, head]

    #--- Create the 3D unit sphere
    def create_sphere():
        #--- Create unit sphere
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)

        sphere = go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.2,
            showscale=False,
            colorscale='Greys',
            hoverinfo='skip'
        )

        #--- Grid lines
        lat_lines = []
        lon_lines = []
        theta = np.linspace(0, 2 * np.pi, 100)
        phi_vals = np.linspace(0.1, np.pi - 0.1, 8)
        phi_vals = np.insert(phi_vals,0,np.pi/2) # Add equator
        theta_vals = np.linspace(0, 2 * np.pi, 12)

        for phi in phi_vals:
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.full_like(theta, np.cos(phi))
            lat_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                        line=dict(color='gray', width=1),
                                        showlegend=False, hoverinfo='skip'))
        for theta0 in theta_vals:
            phi = np.linspace(0, np.pi, 100)
            x = np.cos(theta0) * np.sin(phi)
            y = np.sin(theta0) * np.sin(phi)
            z = np.cos(phi)
            lon_lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                        line=dict(color='gray', width=1),
                                        showlegend=False, hoverinfo='skip'))
        return sphere, lat_lines, lon_lines
    [sphere, lat_lines, lon_lines] = create_sphere()


    #--- Inertial frame arrows
    origin = np.array([0, 0, 0])
    frame_axes = []
    frame_axes += create_arrow(origin, np.array([1, 0, 0]), color='red', name='e_x',dash = 'dashdot')
    frame_axes += create_arrow(origin, np.array([0, 1, 0]), color='lime', name='e_y',dash = 'dashdot')
    frame_axes += create_arrow(origin, np.array([0, 0, 1]), color='cyan', name='e_z',dash = 'dashdot')

    #--- Load quaternion data
    def readQuat_computeError(df):
        """Read quaternion from simulation data. Compute quaternion error and output figure objects for plotting."""
        att_signals = [col for col in df.columns if col.startswith('att_')]
        attdes_signals = [col for col in df.columns if (col.startswith('att_') and col.endswith('des'))]
        quats = df[att_signals].to_numpy()
        quat_des = df[attdes_signals].to_numpy()

        #--- Compute quaternion error (q_err = q_des_conj * q)
        q_err = []
        for q, qd in zip(quats, quat_des):
            # scipy expects [x, y, z, w]
            qd_scipy = [qd[0], qd[1], qd[2], qd[3]]
            q_scipy = [q[0],q[1], q[2], q[3]]
            qd_conj = R.from_quat(qd_scipy).inv()
            q_err_rot = qd_conj * R.from_quat(q_scipy)
            q_err.append(q_err_rot.as_quat())  # [x, y, z, w]
        q_err = np.array(q_err)
        quats = q_err
        x_b = np.array([1, 0, 0])
        y_b = np.array([0, 1, 0])
        z_b = np.array([0, 0, 1])

        x_hist, y_hist, z_hist = [], [], []
        for q in quats:
            rot = R.from_quat([q[0],q[1], q[2], q[3]])  # scipy expects [x, y, z, w]
            x_hist.append(rot.apply(x_b))
            y_hist.append(rot.apply(y_b))
            z_hist.append(rot.apply(z_b))

        x_hist = np.array(x_hist)
        y_hist = np.array(y_hist)
        z_hist = np.array(z_hist)

        #--- Axis trajectories
        trace_x = go.Scatter3d(x=x_hist[:,0], y=x_hist[:,1], z=x_hist[:,2],
            mode='lines',
            line=dict(width=4, color='red'),
            name='X Axis'
        )
        trace_y = go.Scatter3d(x=y_hist[:,0], y=y_hist[:,1], z=y_hist[:,2],
            mode='lines',
            line=dict(width=4, color='lime'),
            name='Y Axis'
        )
        trace_z = go.Scatter3d(x=z_hist[:,0], y=z_hist[:,1], z=z_hist[:,2],
            mode='lines',
            line=dict(width=4, color='cyan'),
            name='Z Axis'
        )
        #--- Start and end markers
        markers = [
            go.Scatter3d(x=[x_hist[0,0]], y=[x_hist[0,1]], z=[x_hist[0,2]], mode='markers',
                        marker=dict(size=6, symbol='circle-open', color='red'), name='X Start'),
            go.Scatter3d(x=[x_hist[-1,0]], y=[x_hist[-1,1]], z=[x_hist[-1,2]], mode='markers',
                        marker=dict(size=6, symbol='circle', color='red'), name='X End'),
            go.Scatter3d(x=[y_hist[0,0]], y=[y_hist[0,1]], z=[y_hist[0,2]], mode='markers',
                        marker=dict(size=6, symbol='circle-open', color='lime'), name='Y Start'),
            go.Scatter3d(x=[y_hist[-1,0]], y=[y_hist[-1,1]], z=[y_hist[-1,2]], mode='markers',
                        marker=dict(size=6, symbol='circle', color='lime'), name='Y End'),
            go.Scatter3d(x=[z_hist[0,0]], y=[z_hist[0,1]], z=[z_hist[0,2]], mode='markers',
                        marker=dict(size=6, symbol='circle-open', color='cyan'), name='Z Start'),
            go.Scatter3d(x=[z_hist[-1,0]], y=[z_hist[-1,1]], z=[z_hist[-1,2]], mode='markers',
                        marker=dict(size=6, symbol='circle', color='cyan'), name='Z End'),
        ]
        #--- Body frame arrows
        origin = np.array([0, 0, 0])
        b_frame_axes = []
        b_frame_axes += create_arrow(origin, np.array([x_hist[-1,0], x_hist[-1,1], x_hist[-1,2]]), color='red', name='X_body')
        b_frame_axes += create_arrow(origin, np.array([y_hist[-1,0], y_hist[-1,1], y_hist[-1,2]]), color='lime', name='Y_body')
        b_frame_axes += create_arrow(origin, np.array([z_hist[-1,0], z_hist[-1,1], z_hist[-1,2]]), color='cyan', name='Z_body')
        return trace_x, trace_y, trace_z, markers, b_frame_axes
    df = pd.read_csv("Simulation/tfc_single"+str(data_value)+"_simout.csv")
    [trace_x, trace_y, trace_z, markers, b_frame_axes] = readQuat_computeError(df)

    #--- Final plot layout
    fig = go.Figure(data=[sphere, trace_x, trace_y, trace_z] +  b_frame_axes  + frame_axes + markers + lat_lines + lon_lines)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.2,1.2], showgrid=False, zeroline=False),
            yaxis=dict(range=[-1.2,1.2], showgrid=False, zeroline=False),
            zaxis=dict(range=[-1.2,1.2], showgrid=False, zeroline=False),
        ),
        template="plotly_dark",
        title='Attitude Trajectories on the Unit Sphere',
        legend=dict(x=0.02, y=0.98, font=dict(color='white'))
    )
    return fig

def update_custom_graph(data_value):
    return get_custom_figure(data_value)

if __name__ == '__main__':
    app.run(debug=True)