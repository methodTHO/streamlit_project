#!/usr/bin/env python3
"""Final verification of Robot Tour TUI implementation"""

import sys
import os

print("=" * 70)
print("FINAL VERIFICATION - Robot Tour TUI Application")
print("=" * 70)

# Check files exist
required_files = [
    'robot_tour_tui.py',
    'test_robot_tour_tui.py', 
    'README_TUI.md'
]

print("\n1. Checking files...")
for f in required_files:
    exists = os.path.exists(f)
    status = "✓" if exists else "✗"
    size = f"{os.path.getsize(f)} bytes" if exists else "N/A"
    print(f"   {status} {f:30s} ({size})")

# Import and verify core module
print("\n2. Importing modules...")
try:
    from robot_tour_tui import (
        RobotTourApp, AppState, Waypoint,
        _to_roman, _rotate_vector
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test core functions
print("\n3. Testing core functions...")
try:
    assert _to_roman(5) == 'V'
    assert _rotate_vector(1.0, 0.0, 0) == (1.0, 0.0)
    print("   ✓ Helper functions working")
except Exception as e:
    print(f"   ✗ Function test failed: {e}")
    sys.exit(1)

# Test data models
print("\n4. Testing data models...")
try:
    wp = Waypoint(x=1.5, y=2.5, wp_type='F')
    assert wp.x == 1.5 and wp.y == 2.5 and wp.wp_type == 'F'
    print("   ✓ Waypoint model working")
except Exception as e:
    print(f"   ✗ Waypoint test failed: {e}")
    sys.exit(1)

# Test AppState
print("\n5. Testing AppState...")
try:
    state = AppState()
    state.track_orientation = '4x5'
    assert state.get_x_labels() == ['A', 'B', 'C', 'D']
    assert state.get_y_labels() == ['1', '2', '3', '4', '5']
    state.reset()
    assert state.track_orientation is None
    print("   ✓ AppState model working")
except Exception as e:
    print(f"   ✗ AppState test failed: {e}")
    sys.exit(1)

# Test app instantiation
print("\n6. Testing RobotTourApp...")
try:
    app = RobotTourApp()
    assert app.state is not None
    assert isinstance(app.state, AppState)
    print("   ✓ RobotTourApp instantiation working")
except Exception as e:
    print(f"   ✗ App instantiation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VERIFICATION CHECKS PASSED!")
print("=" * 70)
print("\nThe Robot Tour TUI application is ready to use!")
print("\nRun it with: python3 robot_tour_tui.py")
print("\nTest suite: python3 test_robot_tour_tui.py")
print("Documentation: README_TUI.md")
