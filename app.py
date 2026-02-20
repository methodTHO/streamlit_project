import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import math
import time
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title='Robot Tour', layout='wide')

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
if 'animate_robot' not in st.session_state:
    st.session_state.animate_robot = False
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
                      line=dict(color='#8B4513', width=8))

    # Draw placed bottles as blue circles at the midpoint of each dashed segment
    RAD = 0.1
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
        marker=dict(size=40, color='rgba(0,0,0,0)', symbol='circle',
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
            marker=dict(symbol=_arrow_symbol, size=30, color='#00cc44',
                        line=dict(color='white', width=2)),
            text=['(0,0)'],
            textposition=_textpos,
            textfont=dict(size=16, color='#00cc44', family='monospace'),
            showlegend=False, hoverinfo='skip',
        ))

    # Invisible cell centers — clickable in Target Point mode (transformed)
    tp = st.session_state.target_point
    g_plot = [_transform_point(x, y, rot) for x, y in gcd]
    gx = [p[0] for p in g_plot]
    gy = [p[1] for p in g_plot]
    cell_opacity = 0.0 if mode not in ('Target Point', 'Gates', 'Move Forward', 'Move Backward', 'Move to Gate') else 0.01
    cell_hover = 'Click to place target' if mode == 'Target Point' else ('Click to toggle gate' if mode == 'Gates' else ('Click to add waypoint' if mode in ('Move Forward', 'Move Backward', 'Move to Gate') else None))
    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=55, color='rgba(0,0,0,0)', symbol='square',
                    opacity=cell_opacity),
        hovertemplate=f'{cell_hover}<extra></extra>' if cell_hover else None,
        hoverinfo='skip' if mode not in ('Target Point', 'Gates', 'Move Forward', 'Move Backward', 'Move to Gate') else None,
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
            textfont=dict(size=70, color='rgba(148,0,211,0.25)', family='Arial Black'),
            showlegend=False, hoverinfo='skip',
        ))

    # Draw path lines: start → wp1 → wp2 → …
    route_pts = st.session_state.route_points
    if route_pts:
        # Build list of (rounded_key, transformed_coord, seg_type)
        # Round coords to 2dp to ensure consistent key matching across bcd/gcd sources
        def _seg_key(ax, ay):
            return (round(float(ax), 2), round(float(ay), 2))

        raw_coords = []
        if sp:
            raw_coords.append((_seg_key(sp[0], sp[1]), _transform_point(sp[0], sp[1], rot), None))
        for rp in route_pts:
            wp_type = rp[2] if len(rp) > 2 else 'F'
            raw_coords.append((_seg_key(rp[0], rp[1]), _transform_point(rp[0], rp[1], rot), wp_type))

        # Draw backward segments first (underneath) as solid red with direction arrows
        for seg_i in range(1, len(raw_coords)):
            if (raw_coords[seg_i][2] or 'F') == 'B':
                x0, y0 = raw_coords[seg_i - 1][1]
                x1, y1 = raw_coords[seg_i][1]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                # arrow tip slightly past midpoint, tail slightly before
                off = 0.12
                dx, dy = x1 - x0, y1 - y0
                length = math.sqrt(dx*dx + dy*dy) or 1
                ux, uy = dx/length, dy/length
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(color='rgba(204,0,0,0.9)', width=7, dash='solid'),
                    showlegend=False, hoverinfo='skip',
                ))
                fig.add_annotation(
                    x=mx + ux*off, y=my + uy*off,
                    ax=mx - ux*off, ay=my - uy*off,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3,
                    arrowcolor='rgba(204,0,0,0.9)', text='',
                )

        # Draw forward/gate segments as dotted orange with direction arrows
        for seg_i in range(1, len(raw_coords)):
            seg_type = raw_coords[seg_i][2] or 'F'
            if seg_type in ('F', 'G'):
                x0, y0 = raw_coords[seg_i - 1][1]
                x1, y1 = raw_coords[seg_i][1]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                off = 0.12
                dx, dy = x1 - x0, y1 - y0
                length = math.sqrt(dx*dx + dy*dy) or 1
                ux, uy = dx/length, dy/length
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(color='#FF8C00', width=3, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                ))
                fig.add_annotation(
                    x=mx + ux*off, y=my + uy*off,
                    ax=mx - ux*off, ay=my - uy*off,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                    arrowcolor='#FF8C00', text='',
                )

    # Draw route points grouped by cell, split by type for coloring
    route_by_cell = {}
    for i, rp in enumerate(st.session_state.route_points):
        key = (rp[0], rp[1])
        if key not in route_by_cell:
            route_by_cell[key] = {'F': [], 'B': [], 'G': []}
        wp_type = rp[2] if len(rp) > 2 else 'F'
        route_by_cell[key][wp_type].append(i + 1)
    for (rpx_, rpy_), by_type in route_by_cell.items():
        rpx_t, rpy_t = _transform_point(rpx_, rpy_, rot)
        has_both = bool(by_type['F'] or by_type['G']) and bool(by_type['B'])
        # slight vertical offset (in data coords) when both types share a cell
        offsets = {'F': 0.15 if has_both else 0.0, 'B': -0.15 if has_both else 0.0, 'G': 0.15 if has_both else 0.0}
        colors = {'F': '#FF8C00', 'B': '#cc0000', 'G': '#20B2AA'}
        for t in ('F', 'G', 'B'):
            if not by_type[t]:
                continue
            nums = by_type[t]
            rows = [','.join(str(n) for n in nums[j:j+3]) for j in range(0, len(nums), 3)]
            label = '<br>'.join(rows)
            fig.add_trace(go.Scatter(
                x=[rpx_t], y=[rpy_t + offsets[t]], mode='text',
                text=[label],
                textposition='middle center',
                textfont=dict(size=20, color=colors[t], family='Arial Black'),
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
            marker=dict(size=40, color='rgba(0,0,0,0.01)', symbol='square'),
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
            marker=dict(symbol='x', size=26, color='rgba(204,0,0,0.35)',
                        line=dict(color='rgba(204,0,0,0.35)', width=4)),
            text=[target_label],
            textposition='top center',
            textfont=dict(size=16, color='rgba(204,0,0,0.35)', family='monospace'),
            showlegend=False, hoverinfo='skip',
        ))



    fig.update_layout(
        height=750,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#f5f0e8',
        paper_bgcolor='white',
        xaxis=dict(range=[-0.3, COLS + 0.3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-0.3, ROWS + 0.3], showgrid=False, zeroline=False,
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

MODES = ['Starting Point', 'Target Point', 'Gates', 'Obstacles', 'Bottles', 'Move Forward', 'Move Backward', 'Move to Gate']

# Sidebar
with st.sidebar:
    if st.session_state.mode not in MODES:
        st.session_state.mode = MODES[0]
    selected_mode = st.radio(
        'Mode', MODES,
        index=MODES.index(st.session_state.mode),
    )
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        st.session_state.mode_change_count += 1
        st.session_state.animate_robot = False
        st.rerun()
    st.divider()
    if st.button('Undo Move', use_container_width=True):
        if st.session_state.route_points:
            st.session_state.route_points.pop()
            st.session_state.route_click_count += 1
            st.rerun()
    if st.button('Clear Route', use_container_width=True):
        st.session_state.route_points = []
        st.session_state.route_click_count += 1
        st.rerun()
    if st.button('Verify Route On/Off', use_container_width=True):
        st.session_state.animate_robot = not st.session_state.animate_robot
        st.rerun()
    if st.button('Reset', use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
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
    elif st.session_state.mode == 'Move to Gate':
        st.write('Click any square to add a Move to Gate waypoint (offset 0.2 from center toward approach direction).')
    elif st.session_state.mode == 'Gates':
        st.write('Click any square to add or remove a gate.')
    elif st.session_state.mode == 'Obstacles':
        st.write('Click any dashed line to place or remove an obstacle.')
    else:
        st.write('Click any dashed line to place or remove a water bottle.')

st.markdown("""<style>
    .block-container { padding-top: 2rem !important; }
</style>""", unsafe_allow_html=True)

st.markdown('## Robot Tour')

def build_animation_figure(bcd, gcd, ocd):
    """Build Plotly figure with animation frames for robot movement."""
    fig = build_figure(bcd, gcd, ocd)
    rot = _rotation_for_start()
    sp = st.session_state.start_point
    route_pts = st.session_state.route_points

    path = []
    if sp:
        path.append(_transform_point(sp[0], sp[1], rot))
    for rp in route_pts:
        path.append(_transform_point(rp[0], rp[1], rot))

    if len(path) < 2:
        return fig

    N_STEPS = 15

    # Per-frame arrays
    frame_xs, frame_ys, frame_angles = [], [], []
    frame_seg_types = []  # 'F', 'B', or 'G' for each frame
    frame_uxs, frame_uys = [], []  # arrow nose unit vector (data coords)

    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_type = route_pts[i][2] if i < len(route_pts) and len(route_pts[i]) > 2 else 'F'
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        travel_ux, travel_uy = dx / length, dy / length
        # Arrow nose faces forward on F/G, backward on B
        nose_ux = travel_ux if seg_type != 'B' else -travel_ux
        nose_uy = travel_uy if seg_type != 'B' else -travel_uy
        base_angle = -math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0
        base_angle = (base_angle + 180) % 360 if seg_type == 'B' else base_angle
        angle = (base_angle + 90) % 360
        for s in range(N_STEPS):
            t = s / N_STEPS
            frame_xs.append(x0 + t * (x1 - x0))
            frame_ys.append(y0 + t * (y1 - y0))
            frame_angles.append(angle)
            frame_seg_types.append(seg_type)
            frame_uxs.append(nose_ux)
            frame_uys.append(nose_uy)

    frame_xs.append(path[-1][0])
    frame_ys.append(path[-1][1])
    frame_angles.append(frame_angles[-1] if frame_angles else 0)
    frame_seg_types.append(frame_seg_types[-1] if frame_seg_types else 'F')
    frame_uxs.append(frame_uxs[-1] if frame_uxs else 0)
    frame_uys.append(frame_uys[-1] if frame_uys else 0)

    # ── Bottle animation setup ────────────────────────────────────────────────
    # ARROW_TIP: how far the arrow tip extends from the robot centre in data units.
    # Size-45 arrow at ~138 px/unit → tip ≈ 45/2 / 138 ≈ 0.16 units; use 0.2 as safe value.
    ARROW_TIP = 0.16     # size-45px arrow: tip is ~22.5px from centre; at ~140px/unit = 0.16 units
    PICKUP_RADIUS = 0.2    # must be > step size (~0.067/unit) to guarantee a hit

    # Compute bottle positions in transformed (screen) coords
    bottle_screen = []
    for bot in st.session_state.bottles:
        orient, pos, cell_idx = bot[0], bot[1], bot[2]
        if orient == 'v':
            bx, by = _transform_point(pos, cell_idx + 0.5, rot)
        else:
            bx, by = _transform_point(cell_idx + 0.5, pos, rot)
        bottle_screen.append((bx, by))

    n_bottles = len(bottle_screen)
    bottle_pickup_frame = [None] * n_bottles   # frame index when tip first touches
    bottle_drop_frame   = [None] * n_bottles   # frame index when backward begins
    bottle_drop_pos     = [None] * n_bottles   # (x, y) where it gets left

    for f in range(len(frame_xs)):
        seg_type = frame_seg_types[f]
        rx, ry = frame_xs[f], frame_ys[f]
        tip_x = rx + frame_uxs[f] * ARROW_TIP
        tip_y = ry + frame_uys[f] * ARROW_TIP
        for b_idx, (bx, by) in enumerate(bottle_screen):
            if bottle_pickup_frame[b_idx] is None:
                # Only pick up during forward / gate moves
                if seg_type in ('F', 'G'):
                    if math.sqrt((tip_x - bx) ** 2 + (tip_y - by) ** 2) < PICKUP_RADIUS:
                        bottle_pickup_frame[b_idx] = f
            elif bottle_drop_frame[b_idx] is None:
                if seg_type == 'B':
                    bottle_drop_frame[b_idx] = f
                    # Drop at the waypoint (robot centre at transition = path[i])
                    bottle_drop_pos[b_idx] = (frame_xs[f], frame_ys[f])

    def _bottle_positions_at(f):
        xs, ys = [], []
        tip_x = frame_xs[f] + frame_uxs[f] * ARROW_TIP
        tip_y = frame_ys[f] + frame_uys[f] * ARROW_TIP
        for b_idx, (bx, by) in enumerate(bottle_screen):
            pf = bottle_pickup_frame[b_idx]
            df = bottle_drop_frame[b_idx]
            if pf is None or f < pf:
                # Not yet picked up — stay at original position
                xs.append(bx); ys.append(by)
            elif df is not None and f >= df:
                # Dropped — stay at drop position
                xs.append(bottle_drop_pos[b_idx][0])
                ys.append(bottle_drop_pos[b_idx][1])
            else:
                # Carried — ride at arrow tip
                xs.append(tip_x); ys.append(tip_y)
        return xs, ys

    # Remove static bottle circles from layout so they don't show during animation
    if n_bottles:
        fig.layout.shapes = tuple(
            s for s in fig.layout.shapes if s.type != 'circle'
        )

    # ── Traces ───────────────────────────────────────────────────────────────
    init_bxs, init_bys = _bottle_positions_at(0)
    bottle_trace = go.Scatter(
        x=init_bxs, y=init_bys,
        mode='markers',
        marker=dict(symbol='circle', size=25, color='#1E90FF',
                    line=dict(color='white', width=1.5)),
        showlegend=False, hoverinfo='skip', name='bottles'
    )
    fig.add_trace(bottle_trace)
    bottle_trace_idx = len(fig.data) - 1

    robot_trace = go.Scatter(
        x=[frame_xs[0]], y=[frame_ys[0]],
        mode='markers',
        marker=dict(symbol='arrow', size=45, color='white',
                    line=dict(color='#222222', width=2),
                    angle=frame_angles[0], angleref='up'),
        showlegend=False, hoverinfo='skip', name='robot'
    )
    fig.add_trace(robot_trace)
    robot_trace_idx = len(fig.data) - 1

    def _make_frame_data(f):
        bxs, bys = _bottle_positions_at(f)
        return [
            go.Scatter(x=bxs, y=bys, mode='markers',
                       marker=dict(symbol='circle', size=25, color='#1E90FF',
                                   line=dict(color='white', width=1.5))),
            go.Scatter(x=[frame_xs[f]], y=[frame_ys[f]], mode='markers',
                       marker=dict(symbol='arrow', size=45, color='white',
                                   line=dict(color='#222222', width=2),
                                   angle=frame_angles[f], angleref='up')),
        ]

    N_LOOPS = 30
    PAUSE_FRAMES = 25  # 1 s pause at end of each loop
    base_frame_data = [_make_frame_data(f) for f in range(len(frame_xs))]
    n = len(base_frame_data)
    last_fd = base_frame_data[-1]
    frames = []
    frame_idx = 0
    for loop in range(N_LOOPS):
        for i in range(n):
            frames.append(go.Frame(
                data=base_frame_data[i],
                traces=[bottle_trace_idx, robot_trace_idx],
                name=str(frame_idx)
            ))
            frame_idx += 1
        for _ in range(PAUSE_FRAMES):
            frames.append(go.Frame(
                data=last_fd,
                traces=[bottle_trace_idx, robot_trace_idx],
                name=str(frame_idx)
            ))
            frame_idx += 1
    fig.frames = frames

    return fig


def render_chart():
    mode = st.session_state.mode
    fig = build_figure(bcd, gcd, ocd)
    if mode == 'Obstacles':
        _click_ver = st.session_state.obs_click_count
    elif mode == 'Bottles':
        _click_ver = st.session_state.bottle_click_count
    elif mode == 'Gates':
        _click_ver = st.session_state.gate_click_count
    elif mode in ('Move Forward', 'Move Backward', 'Move to Gate'):
        _click_ver = st.session_state.route_click_count
    else:
        _click_ver = ''
    chart_key = f'chart_{mode}_{_click_ver}_{st.session_state.mode_change_count}'
    clicked = plotly_events(fig, click_event=True, key=chart_key, override_height=775)

    if clicked:
        pt = clicked[0]
        curve = pt.get('curveNumber', -1)
        idx = pt.get('pointNumber', -1)
        has_arrow = st.session_state.start_point is not None

        # cell_curve accounts for boundary trace (1) + optional arrow trace
        # Gates, path line (if any), and route label traces follow cell trace
        n_gates = len(st.session_state.gates)
        n_route_traces = len(set((r[0], r[1], r[2] if len(r) > 2 else 'F') for r in st.session_state.route_points))
        has_path_line = len(st.session_state.route_points) if st.session_state.route_points else 0
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
        elif mode in ('Move Forward', 'Move Backward', 'Move to Gate') and curve == cell_curve and 0 <= idx < len(gcd):
            cd = gcd[idx]
            if mode == 'Move to Gate':
                # Compute direction from previous point to this cell center, snap to cardinal
                prev = st.session_state.route_points[-1][:2] if st.session_state.route_points else \
                       (st.session_state.start_point[:2] if st.session_state.start_point else None)
                if prev:
                    ddx, ddy = cd[0] - prev[0], cd[1] - prev[1]
                    if abs(ddx) >= abs(ddy) and ddx != 0:
                        ndx, ndy = (1 if ddx > 0 else -1), 0
                    elif ddy != 0:
                        ndx, ndy = 0, (1 if ddy > 0 else -1)
                    else:
                        ndx, ndy = 0, 0
                    ox, oy = cd[0] - 0.2 * ndx, cd[1] - 0.2 * ndy
                else:
                    ox, oy = cd[0], cd[1]
                st.session_state.route_points.append([ox, oy, 'G'])
            else:
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
    if st.session_state.animate_robot:
        anim_fig = build_animation_figure(bcd, gcd, ocd)
        st.plotly_chart(anim_fig, use_container_width=True)
        components.html("""
        <script>
        function tryPlay() {
            var divs = window.parent.document.querySelectorAll('.js-plotly-plot');
            if (divs.length > 0) {
                window.parent.Plotly.animate(divs[divs.length - 1], null, {
                    frame: {duration: 40, redraw: true},
                    transition: {duration: 0},
                    mode: 'immediate'
                });
            } else {
                setTimeout(tryPlay, 200);
            }
        }
        setTimeout(tryPlay, 800);
        </script>
        """, height=0)
    else:
        render_chart()
with col_wp:
    # Waypoints
    n_wp = len(st.session_state.route_points)
    st.markdown(f'#### Waypoints [{n_wp}]')
    total_time = st.number_input('Total time (s)', min_value=1, max_value=600, value=60, step=1)
    if n_wp == 0:
        st.caption('None yet.')
    else:
        rot = _rotation_for_start()
        sp = st.session_state.start_point

        # --- First pass: collect per-waypoint data ---
        wp_data = []  # list of dicts: coord, dist, abs_turn, turn_str, wp_type, meta
        prev_rdx, prev_rdy = 0.0, 1.0
        prev_x = sp[0] if sp else None
        prev_y = sp[1] if sp else None
        for rp in st.session_state.route_points:
            wp_type = rp[2] if len(rp) > 2 else 'F'
            if sp:
                sx, sy, _ = sp
                dx, dy = rp[0] - sx, rp[1] - sy
                rdx, rdy = _rotate_vector(dx, dy, rot)
                def _fmt(v):
                    return str(int(v)) if v == int(v) else f'{v:.1f}'
                coord = f'({_fmt(rdx)}, {_fmt(rdy)})'
                seg_dx = rp[0] - prev_x
                seg_dy = rp[1] - prev_y
                dist = math.hypot(seg_dx, seg_dy)
                dist_str = str(int(dist)) if dist == int(dist) else f'{dist:.2f}'
                seg_rdx, seg_rdy = _rotate_vector(seg_dx, seg_dy, rot)
                if wp_type == 'B':
                    turn_str = '0° S'
                    abs_turn = 0.0
                elif abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                    curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                    prev_angle = math.degrees(math.atan2(prev_rdy, prev_rdx))
                    turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                    t = round(turn)
                    dir_label = 'S' if t == 0 else ('R' if t > 0 else 'L')
                    turn_str = f'{t}° {dir_label}'
                    abs_turn = abs(turn)
                    prev_rdx, prev_rdy = seg_rdx, seg_rdy
                else:
                    turn_str = '—'
                    abs_turn = 0.0
                prev_x, prev_y = rp[0], rp[1]
            else:
                coord = f'({rp[0]}, {rp[1]})'
                dist, dist_str, turn_str, abs_turn = 0.0, '?', '', 0.0
            wp_data.append(dict(coord=coord, dist=dist, dist_str=dist_str,
                                abs_turn=abs_turn, turn_str=turn_str, wp_type=wp_type))

        # --- Time allocation: weight = distance + turn_penalty ---
        # Each 90° of turn is treated as equivalent to 1 extra square
        weights = [d['dist'] + d['abs_turn'] / 90.0 for d in wp_data]
        total_weight = sum(weights) or 1.0
        alloc_times = [w / total_weight * total_time for w in weights]

        # Cumulative timestamps so last waypoint = exactly total_time
        cumulative = []
        running = 0.0
        for i, alloc in enumerate(alloc_times):
            running += alloc
            # Force last waypoint to exact total_time to avoid float drift
            cumulative.append(total_time if i == len(alloc_times) - 1 else running)

        # --- Render lines ---
        lines = []
        for i, (d, seg_t, cum_t) in enumerate(zip(wp_data, alloc_times, cumulative)):
            seg_str = f'{seg_t:.1f}s'
            cum_str = f'{cum_t:.1f}s'
            meta = f'{d["turn_str"]} / {d["dist_str"]}sq / {seg_str} (@ {cum_str})'
            if d['wp_type'] == 'B':
                lines.append(f'{i + 1}. <b><span style="color:red">B</span></b> {d["coord"]} / {meta}')
            elif d['wp_type'] == 'G':
                lines.append(f'{i + 1}. <span style="color:#20B2AA">G</span> {d["coord"]} / {meta}')
            else:
                lines.append(f'{i + 1}. {d["coord"]} / {meta}')
        lines.append(f'<b>Total: {total_time:.1f}s</b>')
        st.markdown('<div style="font-size:1.4em">' + '<br>'.join(lines) + '</div>', unsafe_allow_html=True)

# Arrow shown first; rotate after a brief pause so the user sees the placement
if not st.session_state.get('rotation_ready', True):
    time.sleep(0.4)
    st.session_state.rotation_ready = True
    st.rerun()
