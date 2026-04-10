import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title='Robot Tour Planner', layout='wide')

COLS = 4
ROWS = 5

if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'target_point' not in st.session_state:
    st.session_state.target_point = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'Starting Point'

# --- rotation helpers -------------------------------------------------
# rotation: multiples of 90° CCW needed to bring the start side to the bottom
def _rotation_for_start():
    sp = st.session_state.get('start_point')
    if not sp:
        return 0
    side = sp[2]
    return {'bottom': 0, 'left': 1, 'top': 2, 'right': 3}.get(side, 0)

def _transform_point(x, y, rot):
    """Rotate point (x,y) about the grid center by rot * 90° CCW in *data*
    coordinates. This preserves distances in grid units so squares remain
    square after rotation.
    """
    r = rot % 4
    if r == 0:
        return x, y
    cx, cy = COLS / 2.0, ROWS / 2.0
    dx, dy = x - cx, y - cy
    if r == 1:   # 90° CCW
        nx = cx - dy
        ny = cy + dx
    elif r == 2: # 180°
        nx = cx - dx
        ny = cy - dy
    else:         # 270° CCW
        nx = cx + dy
        ny = cy - dx
    return nx, ny

def _transform_list(coords, rot):
    return [_transform_point(x, y, rot) for x, y in coords]
def _rotate_vector(dx, dy, rot):
    """Rotate a vector (dx,dy) by rot * 90° CCW about the origin."""
    r = rot % 4
    if r == 0:
        return dx, dy
    if r == 1:
        return -dy, dx
    if r == 2:
        return -dx, -dy
    return dy, -dx# -----------------------------------------------------------------------



def build_figure(bcd, gcd):
    fig = go.Figure()

    # Determine display rotation (based on start side)
    rot = _rotation_for_start()

    # Grid background (still full bounding rect)
    fig.add_shape(type='rect', x0=0, y0=0, x1=COLS, y1=ROWS,
                  fillcolor='#f5f0e8', line=dict(width=0), layer='below')

    # Interior dashed lines (transform endpoints)
    for c in range(1, COLS):
        x0, y0 = _transform_point(c, 0, rot)
        x1, y1 = _transform_point(c, ROWS, rot)
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='#aaa', width=1, dash='dash'))
    for r in range(1, ROWS):
        x0, y0 = _transform_point(0, r, rot)
        x1, y1 = _transform_point(COLS, r, rot)
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='#aaa', width=1, dash='dash'))

    # Thick outer border (draw as four transformed edges)
    c0 = _transform_point(0, 0, rot)
    c1 = _transform_point(COLS, 0, rot)
    c2 = _transform_point(COLS, ROWS, rot)
    c3 = _transform_point(0, ROWS, rot)
    fig.add_shape(type='line', x0=c0[0], y0=c0[1], x1=c1[0], y1=c1[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c1[0], y0=c1[1], x1=c2[0], y1=c2[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c2[0], y0=c2[1], x1=c3[0], y1=c3[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c3[0], y0=c3[1], x1=c0[0], y1=c0[1], line=dict(color='#222', width=4))

    sp = st.session_state.start_point

    # Boundary dots — transform for display rotation
    b_plot = [_transform_point(x, y, rot) for x, y, _ in bcd]
    bx = [p[0] for p in b_plot]
    by = [p[1] for p in b_plot]
    if not sp:
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers',
            marker=dict(size=20, color='#00cc44', symbol='circle',
                        line=dict(color='white', width=2)),
            hovertemplate='Click to place start<extra></extra>',
            name='boundary', showlegend=False,
        ))
    else:
        # Invisible hit-targets remain so re-selection still works (but visually hidden)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers',
            marker=dict(size=30, color='rgba(0,0,0,0)', symbol='circle',
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hovertemplate='Click to move start<extra></extra>',
            name='boundary', showlegend=False,
        ))

    # Draw start arrow at transformed location — always point INTO the grid (visual up)
    if sp:
        sx, sy, side = sp
        sx_t, sy_t = _transform_point(sx, sy, rot)
        fig.add_trace(go.Scatter(
            x=[sx_t], y=[sy_t], mode='markers+text',
            marker=dict(symbol='triangle-up', size=22, color='#00cc44',
                        line=dict(color='white', width=2)),
            text=['(0,0)'],
            textposition='bottom center',
            textfont=dict(size=13, color='#00cc44', family='monospace'),
            showlegend=False, hoverinfo='skip',
        ))

    # Invisible cell centers — clickable in Target Point mode (transformed)
    mode = st.session_state.mode
    tp = st.session_state.target_point
    g_plot = [_transform_point(x, y, rot) for x, y in gcd]
    gx = [p[0] for p in g_plot]
    gy = [p[1] for p in g_plot]
    cell_opacity = 0.0 if mode != 'Target Point' else 0.01
    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=40, color='rgba(0,0,0,0)', symbol='square',
                    opacity=cell_opacity),
        hovertemplate='Click to place target<extra></extra>' if mode == 'Target Point' else None,
        hoverinfo='skip' if mode != 'Target Point' else None,
        name='cells', showlegend=False,
    ))

    # Draw target X with coordinates relative to start point (0,0)
    if tp:
        tx, ty = tp
        if sp:
            sx, sy, _ = sp
            def _fmt(v):
                return str(int(v)) if v == int(v) else f'{v:.1f}'
            dx, dy = tx - sx, ty - sy
            rdx, rdy = _rotate_vector(dx, dy, rot)
            target_label = f'({_fmt(rdx)},{_fmt(rdy)})'
        else:
            target_label = 'T'
        tx_t, ty_t = _transform_point(tx, ty, rot)
        fig.add_trace(go.Scatter(
            x=[tx_t], y=[ty_t], mode='markers+text',
            marker=dict(symbol='x', size=20, color='#cc0000',
                        line=dict(color='#cc0000', width=3)),
            text=[target_label],
            textposition='top center',
            textfont=dict(size=13, color='#cc0000', family='monospace'),
            showlegend=False, hoverinfo='skip',
        ))



    fig.update_layout(
        height=600,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor='#f5f0e8',
        paper_bgcolor='white',
        xaxis=dict(range=[-0.6, COLS + 0.3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-0.6, ROWS + 0.3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True,
                   scaleanchor='x', scaleratio=1),
        hovermode='closest',
        dragmode=False,
    )
    return fig


# Build boundary point data
bcd = []
for c in range(COLS):
    bcd.append([c + 0.5, 0.0,         'bottom'])
    bcd.append([c + 0.5, float(ROWS),  'top'])
for r in range(ROWS):
    bcd.append([0.0,         r + 0.5, 'left'])
    bcd.append([float(COLS), r + 0.5, 'right'])

# Build cell center data (for target clicks)
gcd = []
for c in range(COLS):
    for r in range(ROWS):
        gcd.append([c + 0.5, r + 0.5])



MODES = ['Starting Point', 'Target Point']

# Sidebar
with st.sidebar:
    st.session_state.mode = st.radio(
        'Mode', MODES,
        index=MODES.index(st.session_state.mode)
    )
    st.divider()
    if st.session_state.mode == 'Starting Point':
        st.write('Click a green dot on the grid edge.')
    else:
        st.write('Click any square to place the target.')

st.markdown('### Robot Tour Planner')

fig = build_figure(bcd, gcd)

clicked = plotly_events(fig, click_event=True, key='track_chart', override_height=620)

if clicked:
    pt = clicked[0]
    curve = pt.get('curveNumber', -1)
    idx = pt.get('pointNumber', -1)
    mode = st.session_state.mode
    has_arrow = st.session_state.start_point is not None
    has_target = st.session_state.target_point is not None

    # Compute curve indices dynamically
    cell_curve = 1 + (1 if has_arrow else 0)

    if mode == 'Starting Point' and curve == 0 and 0 <= idx < len(bcd):
        cd = bcd[idx]
        new_sp = (cd[0], cd[1], cd[2])
        if st.session_state.start_point != new_sp:
            st.session_state.start_point = new_sp
            st.rerun()
    elif mode == 'Target Point' and curve == cell_curve and 0 <= idx < len(gcd):
        cd = gcd[idx]
        new_tp = (cd[0], cd[1])
        if st.session_state.target_point != new_tp:
            st.session_state.target_point = new_tp
            st.rerun()

sp = st.session_state.start_point
tp = st.session_state.target_point

def _fmt(v):
    return str(int(v)) if v == int(v) else f'{v:.1f}'

if sp:
    st.success('Start: (0,0)')
else:
    st.info('Select **Starting Point** mode and click a green dot on the grid edge.')
if tp:
    if sp:
        rel_x = _fmt(tp[0] - sp[0])
        rel_y = _fmt(tp[1] - sp[1])
        st.success(f'Target: ({rel_x},{rel_y})')
    else:
        st.success(f'Target: ({tp[0]}, {tp[1]})')
else:
    st.info('Select **Target Point** mode and click any square.')
