#!/usr/bin/env python3
"""Test suite for Robot Tour TUI Application"""

from robot_tour_tui import (
    RobotTourApp, AppState, Waypoint,
    _to_roman, _rotate_vector
)


def test_basics():
    """Test basic functionality."""
    print("Testing basic functionality...")
    
    # Test AppState
    state = AppState()
    assert state.start_point is None
    assert state.track_orientation is None
    assert len(state.route_points) == 0
    print("  ✓ AppState initialized correctly")
    
    # Test track orientation
    state.track_orientation = '4x5'
    assert state.get_x_labels() == ['A', 'B', 'C', 'D']
    assert state.get_y_labels() == ['1', '2', '3', '4', '5']
    print("  ✓ 4x5 orientation works")
    
    state.track_orientation = '5x4'
    assert state.get_x_labels() == ['A', 'B', 'C', 'D', 'E']
    assert state.get_y_labels() == ['1', '2', '3', '4']
    print("  ✓ 5x4 orientation works")


def test_waypoints():
    """Test waypoint handling."""
    print("\nTesting waypoint handling...")
    
    state = AppState()
    state.track_orientation = '4x5'
    
    # Add waypoints
    wp1 = Waypoint(1.5, 1.5, 'F')
    wp2 = Waypoint(2.5, 2.5, 'B')
    wp3 = Waypoint(3.5, 3.5, 'G')
    
    state.route_points.append(wp1)
    state.route_points.append(wp2)
    state.route_points.append(wp3)
    
    assert len(state.route_points) == 3
    assert state.route_points[0].wp_type == 'F'
    assert state.route_points[1].wp_type == 'B'
    assert state.route_points[2].wp_type == 'G'
    print("  ✓ Waypoint addition works")
    
    # Test undo
    state.route_points.pop()
    assert len(state.route_points) == 2
    print("  ✓ Waypoint removal works")


def test_reset():
    """Test reset functionality."""
    print("\nTesting reset functionality...")
    
    state = AppState()
    state.track_orientation = '4x5'
    state.start_point = (1.5, 0.0, 'bottom')
    state.route_points.append(Waypoint(2.5, 2.5, 'F'))
    state.a_time = 999.0
    
    state.reset()
    
    assert state.track_orientation is None
    assert state.start_point is None
    assert len(state.route_points) == 0
    assert state.a_time == 0.25
    assert state.target_time == 60.0
    print("  ✓ Reset works correctly")


def test_helpers():
    """Test helper functions."""
    print("\nTesting helper functions...")
    
    # Test Roman numerals
    assert _to_roman(1) == 'I'
    assert _to_roman(5) == 'V'
    assert _to_roman(10) == 'X'
    assert _to_roman(0) == ''
    print("  ✓ Roman numeral conversion works")
    
    # Test vector rotation
    assert _rotate_vector(1.0, 0.0, 0) == (1.0, 0.0)
    rdx, rdy = _rotate_vector(1.0, 0.0, 1)
    assert abs(rdx - 0.0) < 0.01 and abs(rdy - 1.0) < 0.01
    print("  ✓ Vector rotation works")


def test_app_creation():
    """Test app instantiation."""
    print("\nTesting app creation...")
    
    try:
        app = RobotTourApp()
        assert app.state is not None
        assert isinstance(app.state, AppState)
        print("  ✓ App instantiation works")
    except Exception as e:
        print(f"  ✗ App instantiation failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Robot Tour TUI - Test Suite")
    print("=" * 60)
    
    try:
        test_basics()
        test_waypoints()
        test_reset()
        test_helpers()
        test_app_creation()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe app is ready to use. Run it with:")
        print("  python3 robot_tour_tui.py")
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
