import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import math
import time
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title='Robot Tour', layout='wide')

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
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = False
if 'track_orientation' not in st.session_state:
    st.session_state.track_orientation = '4x5'
if 'man_type_gen' not in st.session_state:
    st.session_state.man_type_gen = 0
if 'editing_wp_idx' not in st.session_state:
    st.session_state.editing_wp_idx = None

COLS, ROWS = (4, 5) if st.session_state.track_orientation == '4x5' else (5, 4)

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
            text=['(0.0,0.0)'],
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
                return f'{float(v):.1f}'
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
        height=700,
        autosize=True,
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

MODES = ['Starting Point', 'Target Point', 'Obstacles', 'Bottles', 'Gates', 'Move Forward', 'Move to Gate', 'Move Backward']

# Sidebar
with st.sidebar:
    grid_label = 'Hide Grid' if st.session_state.show_grid else 'Show Grid'
    if st.button(grid_label, width='stretch'):
        st.session_state.show_grid = not st.session_state.show_grid
        st.rerun()
    st.divider()
    if st.session_state.show_grid:
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
    if st.session_state.show_grid and st.button('Undo Move', width='stretch'):
        if st.session_state.route_points:
            st.session_state.route_points.pop()
            st.session_state.route_click_count += 1
            st.rerun()
    if st.session_state.show_grid and st.button('Clear Route', width='stretch'):
        st.session_state.route_points = []
        st.session_state.route_click_count += 1
        st.rerun()
    if st.session_state.show_grid and st.button('Verify Route On/Off', width='stretch'):
        st.session_state.animate_robot = not st.session_state.animate_robot
        st.rerun()
    if st.button('Reset', width='stretch'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    if st.session_state.show_grid:
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
    /* Compact number inputs globally in the time-settings area */
    div[data-testid="stNumberInput"] {
        margin-bottom: -1.1rem !important;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.78em !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stNumberInput"] input {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        font-size: 0.85em !important;
    }
    /* Make target time text input mimic the number‑input square size */
    div[data-testid="stTextInput"] {
        width: 70px !important;
        min-width: 70px !important;
        margin-bottom: 0.2rem !important;  /* keep it tight under its label */
    }
    div[data-testid="stTextInput"] input {
        padding: 4px 6px !important;
        font-size: 0.9em !important;
        width: 100% !important;
        height: 2.2rem !important;
        box-sizing: border-box !important;
    }
    /* Tighten the "Angular / Linear time values" headings */
    div[data-testid="stMarkdown"] p {
        margin-top: 4px !important;
        margin-bottom: 0px !important;
    }

</style>""", unsafe_allow_html=True)

components.html("""
<script>
(function() {
    function _beep() {
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
            osc.start(ctx.currentTime);
            osc.stop(ctx.currentTime + 0.2);
        } catch(e) {}
    }
    function _attach() {
        var doc = window.parent.document;
        doc.querySelectorAll('button').forEach(function(btn) {
            if (!btn._beepAttached) {
                btn._beepAttached = true;
                btn.addEventListener('click', _beep);
            }
        });
    }
    // Attach now and re-attach whenever Streamlit re-renders
    _attach();
    new MutationObserver(_attach).observe(
        window.parent.document.body, {childList: true, subtree: true}
    );
})();
</script>
""", height=0)

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
    clicked = plotly_events(fig, click_event=True, key=chart_key, override_height=900)

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

def _to_roman(n):
    """Convert a positive integer to a Roman numeral string; returns '' for 0."""
    if n <= 0:
        return ''
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for v, s in vals:
        while n >= v:
            result += s
            n -= v
    return result

_RESPONSIVE_JS = """
<script>
(function() {
    function resizeGrid() {
        var P = window.parent;
        if (!P) return;
        var vh = P.innerHeight;
        var vw = P.innerWidth;
        // target 82% of viewport height, clamped between 420 and 1200 px
        var targetH = Math.max(420, Math.min(1200, Math.floor(vh * 0.82)));

        // 1. Resize the plotly_events component iframe
        var compFrames = P.document.querySelectorAll(
            '[data-testid="stCustomComponentV1"] iframe');
        for (var i = 0; i < compFrames.length; i++) {
            var f = compFrames[i];
            f.height = targetH;
            f.style.height = targetH + 'px';
            try {
                var pd = f.contentDocument &&
                         f.contentDocument.querySelector('.js-plotly-plot');
                if (pd && f.contentWindow && f.contentWindow.Plotly) {
                    f.contentWindow.Plotly.relayout(pd, {height: targetH - 10});
                }
            } catch(e) {}
        }

        // 2. Resize static plotly charts (animation mode)
        var mainPlots = P.document.querySelectorAll(
            '[data-testid="stPlotlyChart"] .js-plotly-plot');
        for (var i = 0; i < mainPlots.length; i++) {
            try { P.Plotly.relayout(mainPlots[i], {height: targetH}); }
            catch(e) {}
        }
    }

    // Run immediately and also after DOM settles
    setTimeout(resizeGrid, 300);
    setTimeout(resizeGrid, 800);

    // Re-run on every window resize
    P.removeEventListener('resize', resizeGrid);
    P.addEventListener('resize', resizeGrid);
})();
</script>
"""

# --- Label helpers based on orientation ---
_orient = st.session_state.track_orientation
if _orient == '4x5':
    _x_labels = ['A', 'B', 'C', 'D']   # 4 cols
    _y_labels = ['1', '2', '3', '4', '5']  # 5 rows, 1=bottom
else:  # 5x4
    _x_labels = ['A', 'B', 'C', 'D', 'E']  # 5 cols
    _y_labels = ['1', '2', '3', '4']        # 4 rows, 1=bottom

def _label_to_coord(x_lbl, y_lbl):
    """Convert letter/number labels to internal (x, y) cell-center coords."""
    xi = _x_labels.index(x_lbl)
    yi = _y_labels.index(y_lbl)  # 0 = row 1 = bottom
    return xi + 0.5, yi + 0.5

col_grid, col_wp = st.columns([3, 1])
with col_grid:
    if not st.session_state.show_grid:
        # --- Grid type selector (where the grid graphic would be) ---
        orient_choice = st.radio('Track orientation', ['4x5', '5x4'],
                                 index=0 if st.session_state.track_orientation == '4x5' else 1,
                                 horizontal=True)
        if orient_choice != st.session_state.track_orientation:
            st.session_state.track_orientation = orient_choice
            st.session_state.start_point = None
            st.rerun()
        st.divider()
        # --- Start point selector ---
        st.markdown('**Start Point**')
        _sp_current = st.session_state.start_point
        _sp_cur_col_idx = max(0, min(int(_sp_current[0] - 0.5), len(_x_labels) - 1)) if _sp_current else 0
        _sp1, _sp2 = st.columns([1, 1])
        with _sp1:
            _sp_col = st.selectbox('Column', _x_labels, index=_sp_cur_col_idx, key='sp_col')
        with _sp2:
            st.markdown('<div style="height:1.9em"></div>', unsafe_allow_html=True)
            if st.button('Set Start', key='sp_set'):
                _xi = _x_labels.index(_sp_col)
                st.session_state.prev_rotation = _rotation_for_start()
                st.session_state.start_point = (_xi + 0.5, 0.0, 'bottom')
                st.session_state.rotation_ready = False
                st.rerun()
        if _sp_current:
            _sp_xi = max(0, min(int(_sp_current[0] - 0.5), len(_x_labels) - 1))
            st.caption(f'Current: column {_x_labels[_sp_xi]}')
        else:
            st.caption('No start point set.')
        st.divider()
        # --- Manual waypoint input ---
        st.markdown(f'### Manual Waypoint Input &nbsp; <span style="font-size:0.65em;font-weight:normal;color:#888">({_orient} &nbsp; X={_x_labels[0]}–{_x_labels[-1]}, Y=1–{len(_y_labels)})</span>', unsafe_allow_html=True)
        _mc0, _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns([0.4, 1, 1, 1.5, 1, 1])
        with _mc0:
            st.markdown(f'<div style="height:1.9em"></div><b style="font-size:1.1em">#{len(st.session_state.route_points) + 1}</b>', unsafe_allow_html=True)
        with _mc1:
            _sel_x = st.selectbox('X', _x_labels, key='man_x')
        with _mc2:
            _sel_y = st.selectbox('Y', _y_labels, key='man_y')
        with _mc3:
            _sel_type = st.selectbox('Type', ['Forward', 'Backward', 'Gate'], key=f'man_type_{st.session_state.man_type_gen}')
        with _mc4:
            st.markdown('<div style="height:1.9em"></div>', unsafe_allow_html=True)
            if st.button('Add', key='man_add'):
                _wp_x, _wp_y = _label_to_coord(_sel_x, _sel_y)
                _wp_t = {'Forward': 'F', 'Backward': 'B', 'Gate': 'G'}[_sel_type]
                if _wp_t == 'G':
                    _prev = st.session_state.route_points[-1][:2] if st.session_state.route_points else \
                            (st.session_state.start_point[:2] if st.session_state.start_point else None)
                    if _prev:
                        _ddx, _ddy = _wp_x - _prev[0], _wp_y - _prev[1]
                        if abs(_ddx) >= abs(_ddy) and _ddx != 0:
                            _ndx, _ndy = (1 if _ddx > 0 else -1), 0
                        elif _ddy != 0:
                            _ndx, _ndy = 0, (1 if _ddy > 0 else -1)
                        else:
                            _ndx, _ndy = 0, 0
                        _wp_x, _wp_y = _wp_x - 0.2 * _ndx, _wp_y - 0.2 * _ndy
                st.session_state.route_points.append([_wp_x, _wp_y, _wp_t])
                st.session_state.route_click_count += 1
                st.session_state.man_type_gen += 1
                st.rerun()
        with _mc5:
            st.markdown('<div style="height:1.9em"></div>', unsafe_allow_html=True)
            if st.button('Undo', key='man_undo', disabled=not st.session_state.route_points):
                st.session_state.route_points.pop()
                st.session_state.route_click_count += 1
                st.rerun()
        # Show current waypoints as a simple table
        if st.session_state.route_points:
            st.markdown('**Current waypoints:**')
            for _i, _rp in enumerate(st.session_state.route_points):
                _rx, _ry = _rp[0], _rp[1]
                _rt = _rp[2] if len(_rp) > 2 else 'F'
                _xi = round(_rx - 0.5)
                _yi = round(_ry - 0.5)
                _xlbl = _x_labels[_xi] if 0 <= _xi < len(_x_labels) else f'{_rx:.1f}'
                _ylbl = _y_labels[_yi] if 0 <= _yi < len(_y_labels) else f'{_ry:.1f}'
                _type_name = {'F': 'Forward', 'B': 'Backward', 'G': 'Gate'}.get(_rt, _rt)
                if st.session_state.editing_wp_idx == _i:
                    _ea, _eb, _ec, _ed, _ee = st.columns([1, 1, 1.5, 1, 1])
                    with _ea:
                        _ex_idx = _x_labels.index(_xlbl) if _xlbl in _x_labels else 0
                        _edit_x = st.selectbox('X', _x_labels, index=_ex_idx, key=f'edit_x_{_i}')
                    with _eb:
                        _ey_idx = _y_labels.index(_ylbl) if _ylbl in _y_labels else 0
                        _edit_y = st.selectbox('Y', _y_labels, index=_ey_idx, key=f'edit_y_{_i}')
                    with _ec:
                        _et_idx = ['Forward', 'Backward', 'Gate'].index(_type_name)
                        _edit_type = st.selectbox('Type', ['Forward', 'Backward', 'Gate'], index=_et_idx, key=f'edit_type_{_i}')
                    with _ed:
                        st.markdown('<div style="height:1.9em"></div>', unsafe_allow_html=True)
                        if st.button('Save', key=f'edit_save_{_i}'):
                            _new_x, _new_y = _label_to_coord(_edit_x, _edit_y)
                            _new_t = {'Forward': 'F', 'Backward': 'B', 'Gate': 'G'}[_edit_type]
                            if _new_t == 'G':
                                _prev = st.session_state.route_points[_i - 1][:2] if _i > 0 else \
                                        (st.session_state.start_point[:2] if st.session_state.start_point else None)
                                if _prev:
                                    _ddx, _ddy = _new_x - _prev[0], _new_y - _prev[1]
                                    if abs(_ddx) >= abs(_ddy) and _ddx != 0:
                                        _ndx, _ndy = (1 if _ddx > 0 else -1), 0
                                    elif _ddy != 0:
                                        _ndx, _ndy = 0, (1 if _ddy > 0 else -1)
                                    else:
                                        _ndx, _ndy = 0, 0
                                    _new_x, _new_y = _new_x - 0.2 * _ndx, _new_y - 0.2 * _ndy
                            st.session_state.route_points[_i] = [_new_x, _new_y, _new_t]
                            st.session_state.route_click_count += 1
                            st.session_state.editing_wp_idx = None
                            st.rerun()
                    with _ee:
                        st.markdown('<div style="height:1.9em"></div>', unsafe_allow_html=True)
                        if st.button('Cancel', key=f'edit_cancel_{_i}'):
                            st.session_state.editing_wp_idx = None
                            st.rerun()
                else:
                    _ra, _rb = st.columns([4, 1])
                    with _ra:
                        st.markdown(f'{_i + 1}. **{_xlbl},{_ylbl}** \u2014 {_type_name}')
                    with _rb:
                        if st.button('Edit', key=f'wp_edit_{_i}'):
                            st.session_state.editing_wp_idx = _i
                            st.rerun()
        else:
            st.caption('No waypoints yet.')
    elif st.session_state.animate_robot:
        anim_fig = build_animation_figure(bcd, gcd, ocd)
        st.plotly_chart(anim_fig, width='stretch')
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
        components.html(_RESPONSIVE_JS, height=0)
    else:
        render_chart()
        components.html(_RESPONSIVE_JS, height=0)
with col_wp:
    # Waypoints
    n_wp = len(st.session_state.route_points)
    _tt_raw = st.text_input('Target time', value=str(st.session_state.get('_tt_val', '60')), key='_tt_str')
    try:
        total_time = max(1, int(float(_tt_raw)))
    except (ValueError, TypeError):
        total_time = 60
    st.session_state['_tt_val'] = str(total_time)
    st.markdown('**Angular times:**')
    _ac1, _ac2, _ac3 = st.columns(3)
    with _ac1:
        a_time   = st.number_input('A',   min_value=0.0, max_value=60.0, value=0.25, step=0.1, format='%.2f', key='a_time')
    with _ac2:
        ai_time  = st.number_input('AI',  min_value=0.0, max_value=60.0, value=0.5,  step=0.1, format='%.2f', key='ai_time')
    with _ac3:
        aii_time = st.number_input('AII', min_value=0.0, max_value=60.0, value=0.75, step=0.1, format='%.2f', key='aii_time')
    _angular_time_map = {'': a_time, 'I': ai_time, 'II': aii_time}
    st.markdown('**Linear times:**')
    _lc1, _lc2, _lc3 = st.columns(3)
    with _lc1:
        li_time   = st.number_input('I',   min_value=0.0, max_value=60.0, value=1.0,  step=0.1, format='%.2f', key='li_time')
        liv_time  = st.number_input('IV',  min_value=0.0, max_value=60.0, value=1.75, step=0.1, format='%.2f', key='liv_time')
    with _lc2:
        lii_time  = st.number_input('II',  min_value=0.0, max_value=60.0, value=1.25, step=0.1, format='%.2f', key='lii_time')
        lv_time   = st.number_input('V',   min_value=0.0, max_value=60.0, value=2.0,  step=0.1, format='%.2f', key='lv_time')
    with _lc3:
        liii_time = st.number_input('III', min_value=0.0, max_value=60.0, value=1.5,  step=0.1, format='%.2f', key='liii_time')
    _linear_time_map = {'I': li_time, 'II': lii_time, 'III': liii_time, 'IV': liv_time, 'V': lv_time}
    if n_wp == 0:
        st.markdown('')
        st.markdown(
            f'#### Waypoints '
            f'<span style="font-size:0.6em;font-weight:normal;color:white">'
            f'[WP={n_wp}]</span>',
            unsafe_allow_html=True
        )
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
                    return f'{float(v):.1f}'
                coord = f'({_fmt(rdx)}, {_fmt(rdy)})'
                seg_dx = rp[0] - prev_x
                seg_dy = rp[1] - prev_y
                dist = math.hypot(seg_dx, seg_dy)
                dist_str = str(int(dist)) if dist == int(dist) else f'{dist:.2f}'
                seg_rdx, seg_rdy = _rotate_vector(seg_dx, seg_dy, rot)
                if wp_type == 'B':
                    # Turn is measured from where the robot's back was pointing to the travel direction
                    if abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                        curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                        prev_angle = math.degrees(math.atan2(-prev_rdy, -prev_rdx))  # back of robot
                        turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                        t = round(turn)
                        dir_label = 'S' if t == 0 else ('R' if t > 0 else 'L')
                        turn_str = f'{t}° {dir_label}'
                        abs_turn = abs(t)
                        # After backing, nose faces opposite to travel direction
                        prev_rdx, prev_rdy = -seg_rdx, -seg_rdy
                    else:
                        turn_str = '0° S'
                        abs_turn = 0.0
                elif abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                    curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                    prev_angle = math.degrees(math.atan2(prev_rdy, prev_rdx))
                    turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                    t = round(turn)
                    dir_label = 'S' if t == 0 else ('R' if t > 0 else 'L')
                    turn_str = f'{t}° {dir_label}'
                    abs_turn = abs(t)
                    prev_rdx, prev_rdy = seg_rdx, seg_rdy
                else:
                    turn_str = '—'
                    abs_turn = 0.0
                prev_x, prev_y = rp[0], rp[1]
                linear_roman = _to_roman(math.ceil(dist))
            else:
                coord = f'({float(rp[0]):.1f}, {float(rp[1]):.1f})'
                dist, dist_str, turn_str, abs_turn = 0.0, '?', '', 0.0
                linear_roman = ''
            wp_data.append(dict(coord=coord, dist=dist, dist_str=dist_str,
                                abs_turn=abs_turn, turn_str=turn_str, wp_type=wp_type,
                                linear_roman=linear_roman))

        # --- Turn direction counts ---
        fl = sum(1 for d in wp_data if d['wp_type'] in ('F', 'G') and d['turn_str'].endswith('L'))
        fr = sum(1 for d in wp_data if d['wp_type'] in ('F', 'G') and d['turn_str'].endswith('R'))
        br = sum(1 for d in wp_data if d['wp_type'] == 'B' and d['turn_str'].endswith('R'))
        bl = sum(1 for d in wp_data if d['wp_type'] == 'B' and d['turn_str'].endswith('L'))
        total_l = fl + br
        total_r = fr + bl
        st.markdown('')
        st.markdown(
            f'#### Waypoints '
            f'<span style="font-size:0.6em;font-weight:normal;color:white">'
            f'[WP={n_wp} &nbsp; L={total_l} &nbsp; R={total_r}]</span>',
            unsafe_allow_html=True
        )

        # --- Time allocation: angular time + linear time per waypoint ---
        # (computed here so Total/Diff can be shown right under the heading)
        alloc_times = []
        for d in wp_data:
            ar_suffix = _to_roman(math.ceil(d['abs_turn'] / 90.0))
            lin_t = _linear_time_map.get(d['linear_roman'], 0.0)
            ang_t = _angular_time_map.get(ar_suffix, 0.0)
            alloc_times.append(ang_t + lin_t)

        # Cumulative timestamps
        cumulative = []
        running = 0.0
        for alloc in alloc_times:
            running += alloc
            cumulative.append(running)

        # --- Total / Target / Difference shown right under heading ---
        total_alloc = sum(alloc_times)
        diff = total_alloc - total_time
        diff_sign = '+' if diff >= 0 else ''
        diff_color = '#cc3300' if diff > 0 else ('#00aa44' if diff < 0 else '#888')
        st.markdown(
            f'<div style="font-size:1.0em">'
            f'<b>Total: {total_alloc:.2f}s</b>'
            f'&nbsp;&nbsp;<span style="font-weight:normal;color:#888">Target: {total_time}s</span>'
            f'<br><span style="color:{diff_color}">Difference: {diff_sign}{diff:.2f}s</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # --- Render lines ---
        lines = []
        for i, (d, seg_t, cum_t) in enumerate(zip(wp_data, alloc_times, cumulative)):
            seg_str = f'{seg_t:.2f}s'
            cum_str = f'{cum_t:.2f}s'
            lr = f' ({d["linear_roman"]})' if d['linear_roman'] else ''
            ar_val = _to_roman(math.ceil(d['abs_turn'] / 90.0))
            ar = f' (A{ar_val})'
            move_info = f'{d["turn_str"]}{ar} / {d["dist_str"]}sq{lr}'
            time_info = f'<span style="font-size:0.8em;color:#888">{seg_str} (@ {cum_str})</span>'
            if d['wp_type'] == 'B':
                lines.append(f'{i + 1}. <b><span style="color:red">B</span></b> {d["coord"]} / {move_info}<br>&nbsp;&nbsp;&nbsp;&nbsp;{time_info}')
            elif d['wp_type'] == 'G':
                lines.append(f'{i + 1}. <span style="color:#20B2AA">G</span> {d["coord"]} / {move_info}<br>&nbsp;&nbsp;&nbsp;&nbsp;{time_info}')
            else:
                lines.append(f'{i + 1}. {d["coord"]} / {move_info}<br>&nbsp;&nbsp;&nbsp;&nbsp;{time_info}')
        st.markdown('<div style="font-size:1.0em;">' + '<br>'.join(lines) + '</div>', unsafe_allow_html=True)

# Generated code block — shown below the grid when waypoints exist
if n_wp > 0:
    st.divider()
    st.markdown('**Generated Code**')
    code_lines = []
    for i, (d, seg_t) in enumerate(zip(wp_data, alloc_times)):
        fn = 'Backout' if d['wp_type'] == 'B' else 'GoTo'
        coord_str = d['coord'].strip('()')
        cx_str, cy_str = [s.strip() for s in coord_str.split(',')]
        cx_val = float(cx_str)
        cy_val = float(cy_str)
        linear_roman = d['linear_roman']
        ar_suffix = _to_roman(math.ceil(d['abs_turn'] / 90.0))
        angular_roman = 'A' + ar_suffix
        ang_t = _angular_time_map.get(ar_suffix, 0.0)
        lin_t = _linear_time_map.get(linear_roman, 0.0)
        code_lines.append(f'// Waypoint {i + 1}')
        code_lines.append(f'{fn}(Waypoint {{')
        code_lines.append(f'    x: {cx_val:.1f},')
        code_lines.append(f'    y: {cy_val:.1f},')
        code_lines.append(f'    time_linear: {linear_roman},  // {lin_t:.2f}s')
        code_lines.append(f'    time_angular: {angular_roman},  // {ang_t:.2f}s')
        code_lines.append(f'}}),' )
    st.code('\n'.join(code_lines), language='rust')

# Arrow shown first; rotate after a brief pause so the user sees the placement
if not st.session_state.get('rotation_ready', True):
    time.sleep(0.4)
    st.session_state.rotation_ready = True
    st.rerun()
