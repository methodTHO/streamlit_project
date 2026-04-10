import streamlit as st
import plotly.graph_objects as go
import time
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
if 'obstacles' not in st.session_state:
    st.session_state.obstacles = []  # list of [orient, pos, cell_idx]
if 'obs_click_count' not in st.session_state:
    st.session_state.obs_click_count = 0
if 'bottles' not in st.session_state:
    st.session_state.bottles = []  # list of [orient, pos, cell_idx]
if 'bottle_click_count' not in st.session_state:
    st.session_state.bottle_click_count = 0
if 'gates' not in st.session_state:
    st.session_state.gates = []  # list of [x, y]
if 'gate_click_count' not in st.session_state:
    st.session_state.gate_click_count = 0
if 'route_points' not in st.session_state:
    st.session_state.route_points = []  # list of [x, y]
if 'route_click_count' not in st.session_state:
    st.session_state.route_click_count = 0
if 'mode_change_count' not in st.session_state:
    st.session_state.mode_change_count = 0
if 'rotation_ready' not in st.session_state:
    st.session_state.rotation_ready = True
if 'prev_rotation' not in st.session_state:
    st.session_state.prev_rotation = 0

# --- rotation helpers -------------------------------------------------
# rotation: multiples of 90° CCW needed to bring the start side to the bottom
def _rotation_for_start():
    sp = st.session_state.get('start_point')
    if not sp:
        return 0
    if not st.session_state.get('rotation_ready', True):
        # Step 1: hold at the previous rotation so the grid doesn't jump
        return st.session_state.get('prev_rotation', 0)
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
    return dy, -dx

def _has_obstacle(orient, pos, cell_idx):
    return any(o[0] == orient and o[1] == pos and o[2] == cell_idx for o in st.session_state.obstacles)
# -----------------------------------------------------------------------



def build_figure(bcd, gcd, ocd):
    fig = go.Figure()

    # Determine display rotation (based on start side)
    rot = _rotation_for_start()
    mode = st.session_state.mode

    # Grid background (still full bounding rect)
    fig.add_shape(type='rect', x0=0, y0=0, x1=COLS, y1=ROWS,
                  fillcolor='#f5f0e8', line=dict(width=0), layer='below')

    # Interior dashed lines — draw all, then overlay brown per-cell segments for obstacles
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
    # Draw placed obstacles as solid brown segments — only the one cell's edge
    for obs in st.session_state.obstacles:
        orient, pos, cell_idx = obs[0], obs[1], obs[2]
        if orient == 'v':
            x0, y0 = _transform_point(pos, cell_idx, rot)
            x1, y1 = _transform_point(pos, cell_idx + 1, rot)
        else:
            x0, y0 = _transform_point(cell_idx, pos, rot)
            x1, y1 = _transform_point(cell_idx + 1, pos, rot)
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='#8B4513', width=6))

    # Draw placed bottles as blue circles at the midpoint of each dashed segment
    RAD = 0.11
    for bot in st.session_state.bottles:
        orient, pos, cell_idx = bot[0], bot[1], bot[2]
        if orient == 'v':
            mx, my = _transform_point(pos, cell_idx + 0.5, rot)
        else:
            mx, my = _transform_point(cell_idx + 0.5, pos, rot)
        fig.add_shape(type='circle',
                      x0=mx - RAD, y0=my - RAD, x1=mx + RAD, y1=my + RAD,
                      fillcolor='#1E90FF', line=dict(color='white', width=1.5))
    c0 = _transform_point(0, 0, rot)
    c1 = _transform_point(COLS, 0, rot)
    c2 = _transform_point(COLS, ROWS, rot)
    c3 = _transform_point(0, ROWS, rot)
    fig.add_shape(type='line', x0=c0[0], y0=c0[1], x1=c1[0], y1=c1[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c1[0], y0=c1[1], x1=c2[0], y1=c2[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c2[0], y0=c2[1], x1=c3[0], y1=c3[1], line=dict(color='#222', width=4))
    fig.add_shape(type='line', x0=c3[0], y0=c3[1], x1=c0[0], y1=c0[1], line=dict(color='#222', width=4))

    sp = st.session_state.start_point

    # Boundary hit-targets — always invisible; visible only in Starting Point mode via hover
    b_plot = [_transform_point(x, y, rot) for x, y, _ in bcd]
    bx = [p[0] for p in b_plot]
    by = [p[1] for p in b_plot]
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode='markers',
        marker=dict(size=30, color='rgba(0,0,0,0)', symbol='circle',
                    line=dict(color='rgba(0,0,0,0)', width=0)),
        hovertemplate='Click to place start<extra></extra>' if mode == 'Starting Point' else None,
        hoverinfo='skip' if mode != 'Starting Point' else None,
        name='boundary', showlegend=False,
    ))

    # Draw start arrow at transformed location — always point INTO the grid
    if sp:
        sx, sy, side = sp
        sx_t, sy_t = _transform_point(sx, sy, rot)
        # Outward direction per side in unrotated coords; rotate by current rot to get display direction
        _outward_raw = {'bottom': (0, -1), 'top': (0, 1), 'left': (-1, 0), 'right': (1, 0)}.get(side, (0, -1))
        _ox, _oy = _rotate_vector(_outward_raw[0], _outward_raw[1], rot)
        _textpos = {(0, -1): 'bottom center', (0, 1): 'top center',
                    (-1, 0): 'middle left', (1, 0): 'middle right'}.get(
                        (round(_ox), round(_oy)), 'bottom center')
        if not st.session_state.get('rotation_ready', True):
            # Step 1: grid is at prev_rotation; compute inward direction rotated by prev_rot
            _inward = {'bottom': (0, 1), 'top': (0, -1), 'left': (1, 0), 'right': (-1, 0)}.get(side, (0, 1))
            _rdx, _rdy = _rotate_vector(_inward[0], _inward[1], rot)
            _arrow_symbol = {(0, 1): 'triangle-up', (0, -1): 'triangle-down',
                             (-1, 0): 'triangle-left', (1, 0): 'triangle-right'}.get(
                                 (round(_rdx), round(_rdy)), 'triangle-up')
        else:
            # Step 2: start side is always at the visual bottom after rotation
            _arrow_symbol = 'triangle-up'
        fig.add_trace(go.Scatter(
            x=[sx_t], y=[sy_t], mode='markers+text',
            marker=dict(symbol=_arrow_symbol, size=22, color='#00cc44',
                        line=dict(color='white', width=2)),
            text=['(0,0)'],
            textposition=_textpos,
            textfont=dict(size=13, color='#00cc44', family='monospace'),
            showlegend=False, hoverinfo='skip',
        ))

    # Invisible cell centers — clickable in Target Point mode (transformed)
    tp = st.session_state.target_point
    g_plot = [_transform_point(x, y, rot) for x, y in gcd]
    gx = [p[0] for p in g_plot]
    gy = [p[1] for p in g_plot]
    cell_opacity = 0.0 if mode not in ('Target Point', 'Gates', 'Move Forward', 'Move Backward') else 0.01
    cell_hover = 'Click to place target' if mode == 'Target Point' else ('Click to toggle gate' if mode == 'Gates' else ('Click to add waypoint' if mode in ('Move Forward', 'Move Backward') else None))
    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=40, color='rgba(0,0,0,0)', symbol='square',
                    opacity=cell_opacity),
        hovertemplate=f'{cell_hover}<extra></extra>' if cell_hover else None,
        hoverinfo='skip' if mode not in ('Target Point', 'Gates', 'Move Forward', 'Move Backward') else None,
        name='cells', showlegend=False,
    ))

    # Draw placed gates as capital letters A, B, C…
    for i, gate in enumerate(st.session_state.gates):
        gx_, gy_ = gate[0], gate[1]
        letter = chr(ord('A') + i)
        gx_t, gy_t = _transform_point(gx_, gy_, rot)
        fig.add_trace(go.Scatter(
            x=[gx_t], y=[gy_t], mode='text',
            text=[letter],
            textposition='middle center',
            textfont=dict(size=54, color='rgba(148,0,211,0.25)', family='Arial Black'),
            showlegend=False, hoverinfo='skip',
        ))

    # Draw path lines: start → wp1 → wp2 → …
    route_pts = st.session_state.route_points
    if route_pts:
        path_coords = []
        if sp:
            path_coords.append(_transform_point(sp[0], sp[1], rot))
        for rp in route_pts:
            path_coords.append(_transform_point(rp[0], rp[1], rot))
        if len(path_coords) >= 2:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_coords],
                y=[p[1] for p in path_coords],
                mode='lines',
                line=dict(color='#FF8C00', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ))

    # Draw route points grouped by cell, split by type for coloring
    route_by_cell = {}
    for i, rp in enumerate(st.session_state.route_points):
        key = (rp[0], rp[1])
        if key not in route_by_cell:
            route_by_cell[key] = {'F': [], 'B': []}
        wp_type = rp[2] if len(rp) > 2 else 'F'
        route_by_cell[key][wp_type].append(i + 1)
    for (rpx_, rpy_), by_type in route_by_cell.items():
        rpx_t, rpy_t = _transform_point(rpx_, rpy_, rot)
        has_both = bool(by_type['F']) and bool(by_type['B'])
        # slight vertical offset (in data coords) when both types share a cell
        offsets = {'F': 0.15 if has_both else 0.0, 'B': -0.15 if has_both else 0.0}
        colors = {'F': '#FF8C00', 'B': '#cc0000'}
        for t in ('F', 'B'):
            if not by_type[t]:
                continue
            nums = by_type[t]
            rows = [','.join(str(n) for n in nums[j:j+3]) for j in range(0, len(nums), 3)]
            label = '<br>'.join(rows)
            fig.add_trace(go.Scatter(
                x=[rpx_t], y=[rpy_t + offsets[t]], mode='text',
                text=[label],
                textposition='middle center',
                textfont=dict(size=16, color=colors[t], family='Arial Black'),
                showlegend=False, hoverinfo='skip',
            ))

    # Obstacle hit-targets — only in Obstacles mode
    obs_mode = mode == 'Obstacles'
    bottle_mode = mode == 'Bottles'
    if obs_mode or bottle_mode:
        o_plot = [_transform_point(x, y, rot) for x, y, _, _, _ in ocd]
        ox = [p[0] for p in o_plot]
        oy = [p[1] for p in o_plot]
        label = 'Click to toggle obstacle' if obs_mode else 'Click to place bottle'
        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode='markers',
            marker=dict(size=30, color='rgba(0,0,0,0.01)', symbol='square'),
            hovertemplate=f'{label}<extra></extra>',
            name='line_targets', showlegend=False,
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
        xaxis=dict(range=[-0.9, COLS + 0.9], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-0.9, ROWS + 0.9], showgrid=False, zeroline=False,
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



# Build dashed-line hit-target data (midpoint of each cell segment along every interior line)
ocd = []  # [x, y, orient, pos, cell_idx]
for c in range(1, COLS):
    for r in range(ROWS):
        ocd.append([c, r + 0.5, 'v', c, r])
for r in range(1, ROWS):
    for c in range(COLS):
        ocd.append([c + 0.5, r, 'h', r, c])

MODES = ['Starting Point', 'Target Point', 'Gates', 'Obstacles', 'Bottles', 'Move Forward', 'Move Backward']

def _on_mode_change():
    st.session_state.mode_change_count += 1

# Sidebar
with st.sidebar:
    if st.session_state.mode not in MODES:
        st.session_state.mode = MODES[0]
    st.session_state.mode = st.radio(
        'Mode', MODES,
        index=MODES.index(st.session_state.mode),
        on_change=_on_mode_change
    )
    st.divider()
    if st.button('Reset', use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    if st.button('Clear Route', use_container_width=True):
        st.session_state.route_points = []
        st.session_state.route_click_count += 1
        st.rerun()
    st.divider()
    if st.session_state.mode == 'Starting Point':
        st.write('Click any edge of the grid to place the start.')
    elif st.session_state.mode == 'Target Point':
        st.write('Click any square to place the target.')
    elif st.session_state.mode == 'Move Forward':
        st.write('Click any square to add a Move Forward waypoint.')
    elif st.session_state.mode == 'Move Backward':
        st.write('Click any square to add a Move Backward waypoint.')
    elif st.session_state.mode == 'Gates':
        st.write('Click any square to add or remove a gate.')
    elif st.session_state.mode == 'Obstacles':
        st.write('Click any dashed line to place or remove an obstacle.')
    else:
        st.write('Click any dashed line to place or remove a water bottle.')

st.markdown("""<style>
    .block-container { padding-top: 2rem !important; }
</style>""", unsafe_allow_html=True)

st.markdown('### Robot Tour Planner')

def render_chart():
    mode = st.session_state.mode
    fig = build_figure(bcd, gcd, ocd)
    if mode == 'Obstacles':
        _click_ver = st.session_state.obs_click_count
    elif mode == 'Bottles':
        _click_ver = st.session_state.bottle_click_count
    elif mode == 'Gates':
        _click_ver = st.session_state.gate_click_count
    elif mode in ('Move Forward', 'Move Backward'):
        _click_ver = st.session_state.route_click_count
    else:
        _click_ver = ''
    chart_key = f'chart_{mode}_{_click_ver}_{st.session_state.mode_change_count}'
    clicked = plotly_events(fig, click_event=True, key=chart_key, override_height=620)

    if clicked:
        pt = clicked[0]
        curve = pt.get('curveNumber', -1)
        idx = pt.get('pointNumber', -1)
        has_arrow = st.session_state.start_point is not None

        # cell_curve accounts for boundary trace (1) + optional arrow trace
        # Gates, path line (if any), and route label traces follow cell trace
        n_gates = len(st.session_state.gates)
        n_route_traces = len(set((r[0], r[1], r[2] if len(r) > 2 else 'F') for r in st.session_state.route_points))
        has_path_line = 1 if st.session_state.route_points else 0
        cell_curve = 1 + (1 if has_arrow else 0)
        obstacle_curve = cell_curve + 1 + n_gates + has_path_line + n_route_traces

        if mode == 'Starting Point' and curve == 0 and 0 <= idx < len(bcd):
            cd = bcd[idx]
            new_sp = (cd[0], cd[1], cd[2])
            if st.session_state.start_point != new_sp:
                st.session_state.prev_rotation = _rotation_for_start()  # save current visual rotation
                st.session_state.start_point = new_sp
                st.session_state.rotation_ready = False
                st.rerun()
        elif mode == 'Gates' and curve == cell_curve and 0 <= idx < len(gcd):
            cd = gcd[idx]
            pt_xy = [cd[0], cd[1]]
            new_gates = [g for g in st.session_state.gates if not (g[0] == pt_xy[0] and g[1] == pt_xy[1])]
            if len(new_gates) == len(st.session_state.gates):
                new_gates.append(pt_xy)
            st.session_state.gates = new_gates
            st.session_state.gate_click_count += 1
            st.rerun()
        elif mode in ('Move Forward', 'Move Backward') and curve == cell_curve and 0 <= idx < len(gcd):
            cd = gcd[idx]
            wp_type = 'F' if mode == 'Move Forward' else 'B'
            st.session_state.route_points.append([cd[0], cd[1], wp_type])
            st.session_state.route_click_count += 1
            st.rerun()
        elif mode == 'Obstacles' and curve == obstacle_curve and 0 <= idx < len(ocd):
            orient, pos, cell_idx = ocd[idx][2], ocd[idx][3], ocd[idx][4]
            new_obs = [o for o in st.session_state.obstacles if not (o[0] == orient and o[1] == pos and o[2] == cell_idx)]
            if len(new_obs) == len(st.session_state.obstacles):
                new_obs.append([orient, pos, cell_idx])
            st.session_state.obstacles = new_obs
            st.session_state.obs_click_count += 1
            st.rerun()
        elif mode == 'Bottles' and curve == obstacle_curve and 0 <= idx < len(ocd):
            orient, pos, cell_idx = ocd[idx][2], ocd[idx][3], ocd[idx][4]
            new_bots = [b for b in st.session_state.bottles if not (b[0] == orient and b[1] == pos and b[2] == cell_idx)]
            if len(new_bots) == len(st.session_state.bottles):
                new_bots.append([orient, pos, cell_idx])
            st.session_state.bottles = new_bots
            st.session_state.bottle_click_count += 1
            st.rerun()
        elif mode == 'Target Point' and curve == cell_curve and 0 <= idx < len(gcd):
            cd = gcd[idx]
            new_tp = (cd[0], cd[1])
            if st.session_state.target_point != new_tp:
                st.session_state.target_point = new_tp
                st.rerun()

col_grid, col_wp = st.columns([3, 1])
with col_grid:
    render_chart()
with col_wp:
    # Waypoints
    n_wp = len(st.session_state.route_points)
    st.markdown(f'**Waypoints [{n_wp}]**')
    if n_wp == 0:
        st.caption('None yet.')
    else:
        rot = _rotation_for_start()
        sp = st.session_state.start_point
        lines = []
        for i, rp in enumerate(st.session_state.route_points):
            if sp:
                sx, sy, _ = sp
                dx, dy = rp[0] - sx, rp[1] - sy
                rdx, rdy = _rotate_vector(dx, dy, rot)
                def _fmt(v):
                    return str(int(v)) if v == int(v) else f'{v:.1f}'
                coord = f'({_fmt(rdx)}, {_fmt(rdy)})'
            else:
                coord = f'({rp[0]}, {rp[1]})'
            wp_type = rp[2] if len(rp) > 2 else 'F'
            if wp_type == 'B':
                lines.append(f'{i + 1}. <b><span style="color:red">B</span></b> {coord}')
            else:
                lines.append(f'{i + 1}. {coord}')
        st.markdown('<div style="font-size:1.4em">' + '<br>'.join(lines) + '</div>', unsafe_allow_html=True)

# Arrow shown first; rotate after a brief pause so the user sees the placement
if not st.session_state.get('rotation_ready', True):
    time.sleep(0.4)
    st.session_state.rotation_ready = True
    st.rerun()
