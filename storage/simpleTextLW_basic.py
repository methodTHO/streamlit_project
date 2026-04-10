import streamlit as st
import math
import time

st.set_page_config(page_title='Robot Tour', layout='wide')

# Session state initialization
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'target_point' not in st.session_state:
    st.session_state.target_point = None
if 'route_points' not in st.session_state:
    st.session_state.route_points = []
if 'route_click_count' not in st.session_state:
    st.session_state.route_click_count = 0
if 'track_orientation' not in st.session_state:
    st.session_state.track_orientation = '4x5'
if 'man_type_gen' not in st.session_state:
    st.session_state.man_type_gen = 0
if 'editing_wp_idx' not in st.session_state:
    st.session_state.editing_wp_idx = None

# Helper function for angle calculations
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

# Label helpers based on orientation
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
    yi = _y_labels.index(y_lbl)
    return xi + 0.5, yi + 0.5

st.markdown('## Robot Tour - Basic Mode')

# Main UI
col_grid, col_wp = st.columns([3, 1])

with col_grid:
    # --- Grid type selector ---
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
            st.session_state.start_point = (_xi + 0.5, 0.0, 'bottom')
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
                    st.markdown(f'{_i + 1}. **{_xlbl},{_ylbl}** — {_type_name}')
                with _rb:
                    if st.button('Edit', key=f'wp_edit_{_i}'):
                        st.session_state.editing_wp_idx = _i
                        st.rerun()
    else:
        st.caption('No waypoints yet.')

with col_wp:
    # Waypoints summary
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
        ai_time  = st.number_input('AI',  min_value=0.0, max_value=60.0, value=1.0,  step=0.1, format='%.2f', key='ai_time')
    with _ac3:
        aii_time = st.number_input('AII', min_value=0.0, max_value=60.0, value=1.50, step=0.1, format='%.2f', key='aii_time')
    _angular_time_map = {'': a_time, 'I': ai_time, 'II': aii_time}
    
    st.markdown('**Linear times:**')
    _lc1, _lc2, _lc3 = st.columns(3)
    with _lc1:
        li_time   = st.number_input('I',   min_value=0.0, max_value=60.0, value=2.0,  step=0.1, format='%.2f', key='li_time')
        liv_time  = st.number_input('IV',  min_value=0.0, max_value=60.0, value=4.0, step=0.1, format='%.2f', key='liv_time')
    with _lc2:
        lii_time  = st.number_input('II',  min_value=0.0, max_value=60.0, value=2.0, step=0.1, format='%.2f', key='lii_time')
        lv_time   = st.number_input('V',   min_value=0.0, max_value=60.0, value=5.0,  step=0.1, format='%.2f', key='lv_time')
    with _lc3:
        liii_time = st.number_input('III', min_value=0.0, max_value=60.0, value=3.0,  step=0.1, format='%.2f', key='liii_time')
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
        rot = 0  # No rotation in basic mode for now
        sp = st.session_state.start_point

        # --- First pass: collect per-waypoint data ---
        wp_data = []
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
                    if abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                        curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                        prev_angle = math.degrees(math.atan2(-prev_rdy, -prev_rdx))
                        turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                        t = round(turn)
                        dir_label = 'S' if t == 0 else ('R' if t > 0 else 'L')
                        turn_str = f'{t}° {dir_label}'
                        abs_turn = abs(t)
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

        # --- Time allocation ---
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

        # --- Total / Target / Difference ---
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

# Generated code block
n_wp = len(st.session_state.route_points)
if n_wp > 0:
    st.divider()
    st.markdown('**Generated Code**')
    
    sp = st.session_state.start_point
    rot = 0  # No rotation in basic mode
    
    # Collect waypoint data
    wp_data = []
    prev_rdx, prev_rdy = 0.0, 1.0
    prev_x = sp[0] if sp else None
    prev_y = sp[1] if sp else None
    
    _angular_time_map = {'': 0.25, 'I': 1.0, 'II': 1.50}
    _linear_time_map = {'I': 2.0, 'II': 2.0, 'III': 3.0, 'IV': 4.0, 'V': 5.0}
    
    for rp in st.session_state.route_points:
        wp_type = rp[2] if len(rp) > 2 else 'F'
        if sp:
            sx, sy, _ = sp
            dx, dy = rp[0] - sx, rp[1] - sy
            rdx, rdy = _rotate_vector(dx, dy, rot)
            seg_dx = rp[0] - prev_x
            seg_dy = rp[1] - prev_y
            dist = math.hypot(seg_dx, seg_dy)
            linear_roman = _to_roman(math.ceil(dist))
            
            seg_rdx, seg_rdy = _rotate_vector(seg_dx, seg_dy, rot)
            if wp_type == 'B':
                if abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                    curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                    prev_angle = math.degrees(math.atan2(-prev_rdy, -prev_rdx))
                    turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                    abs_turn = abs(round(turn))
                    prev_rdx, prev_rdy = -seg_rdx, -seg_rdy
                else:
                    abs_turn = 0.0
            elif abs(seg_rdx) > 1e-9 or abs(seg_rdy) > 1e-9:
                curr_angle = math.degrees(math.atan2(seg_rdy, seg_rdx))
                prev_angle = math.degrees(math.atan2(prev_rdy, prev_rdx))
                turn = -((curr_angle - prev_angle + 180) % 360 - 180)
                abs_turn = abs(round(turn))
                prev_rdx, prev_rdy = seg_rdx, seg_rdy
            else:
                abs_turn = 0.0
            
            prev_x, prev_y = rp[0], rp[1]
        else:
            linear_roman = ''
            abs_turn = 0.0
            rdx, rdy = rp[0], rp[1]
        
        wp_data.append(dict(rdx=rdx if sp else rp[0], rdy=rdy if sp else rp[1], 
                           abs_turn=abs_turn, wp_type=wp_type, linear_roman=linear_roman))
    
    code_lines = []
    for i, d in enumerate(wp_data):
        fn = 'Backout' if d['wp_type'] == 'B' else 'GoTo'
        cx_val = d['rdx']
        cy_val = d['rdy']
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

# Reset button
if st.button('Reset All', use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
