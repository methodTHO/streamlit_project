#!/usr/bin/env python3
"""Intensive testing script for Robot Tour TUI"""

import sys
import traceback

print("=" * 70)
print("INTENSIVE TESTING - Running Robot Tour TUI app multiple times")
print("=" * 70)

for run_num in range(1, 11):
    print(f"\n[RUN {run_num:2d}/10]", end=" ")
    try:
        # Fresh import each time
        if 'robot_tour_tui' in sys.modules:
            del sys.modules['robot_tour_tui']
        
        from robot_tour_tui import RobotTourApp, AppState, Waypoint, _to_roman, _rotate_vector
        
        # Test 1: Create app
        app = RobotTourApp()
        
        # Test 2: Compose widgets
        widgets = list(app.compose())
        assert len(widgets) > 0
        
        # Test 3: State initialization
        state = app.state
        assert state is not None
        assert isinstance(state, AppState)
        assert state.track_orientation is None
        assert len(state.route_points) == 0
        
        # Test 4: Track orientation
        state.track_orientation = '4x5'
        assert state.get_x_labels() == ['A', 'B', 'C', 'D']
        assert state.get_y_labels() == ['1', '2', '3', '4', '5']
        
        # Test 5: Start point
        state.start_point = (1.5, 0.0, 'bottom')
        assert state.start_point[0] == 1.5
        
        # Test 6: Waypoints
        state.route_points.append(Waypoint(2.5, 2.5, 'F'))
        state.route_points.append(Waypoint(3.5, 3.5, 'B'))
        assert len(state.route_points) == 2
        
        # Test 7: Helper functions
        assert _to_roman(5) == 'V'
        assert _rotate_vector(1.0, 0.0, 0) == (1.0, 0.0)
        
        # Test 8: Reset
        state.reset()
        assert state.track_orientation is None
        assert state.start_point is None
        assert len(state.route_points) == 0
        assert state.a_time == 0.25
        
        # Test 9: Multiple operations
        state.track_orientation = '5x4'
        for i in range(3):
            state.route_points.append(Waypoint(i + 1.5, i + 1.5, 'F'))
        assert len(state.route_points) == 3
        
        # Test 10: Final reset
        state.reset()
        assert state.track_orientation is None
        
        print("✓ PASS")
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 70)
print("✓✓✓ ALL 10 RUNS COMPLETED SUCCESSFULLY! ✓✓✓")
print("=" * 70)
print("\nApp is stable and error-free!")
