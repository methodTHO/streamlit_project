# Robot Tour TUI (Textual User Interface)

A terminal-based conversion of the Robot Tour Streamlit application using Python's Textual framework.

## Features

The textual app provides the same functionality as the Streamlit app with a TUI interface:

- **Track Orientation Selection**: Choose between 4x5 or 5x4 track layouts
- **Start Point Selection**: Select the starting column (A-D for 4x5, A-E for 5x4)
- **Manual Waypoint Input**: Add waypoints by specifying:
  - X coordinate (column letter)
  - Y coordinate (row number)
  - Waypoint type (Forward, Backward, or Gate)
- **Waypoint Management**: View, edit, and remove waypoints
- **Time Parameters**: Configure:
  - Target time (seconds)
  - Angular times (A, AI, AII)
  - Linear times (I, II, III, IV, V)
- **Summary & Calculations**: View:
  - Total waypoints and time allocations
  - Target vs. actual time
  - Turn counts (left/right)
- **Generated Code**: Automatic Rust-like code generation from waypoint data
- **Reset Functionality**: Clear all settings and return to defaults

## Installation

1. Ensure Python 3.8+ is installed
2. Install Textual (if not already installed):
   ```bash
   pip install textual
   ```

## Usage

Run the application:
```bash
python3 robot_tour_tui.py
```

### Keyboard Shortcuts

- `q` - Quit the application
- `Ctrl+R` - Reset all settings to defaults

### Workflow

1. **Select Track Orientation**: Choose 4x5 or 5x4
2. **Set Start Point**: Select the starting column
3. **Add Waypoints**: Fill in coordinates and type, click "Add Waypoint"
4. **Configure Timing**: Set target time and individual angular/linear times in the right panel
5. **View Results**: Monitor summary calculations and generated code
6. **Reset**: Use the Reset button or Ctrl+R to clear everything

## Testing

Two test suites are included to verify functionality:

### Test Core Logic
```bash
python3 test_robot_tour_tui.py
```

Tests:
- Basic functionality (waypoint addition, Roman numerals, vector rotation)
- Orientation-specific labels
- Waypoint types (Forward, Backward, Gate)
- Default parameter values

### Test UI Components
```bash
python3 test_ui_components.py
```

Tests:
- App instantiation
- State management
- Required methods existence

## File Structure

- `robot_tour_tui.py` - Main textual application (595 lines)
- `test_robot_tour_tui.py` - Core logic tests
- `test_ui_components.py` - UI component tests
- `README_TUI.md` - This file

## Architecture

### Data Models
- `Waypoint`: Represents individual waypoints with position and type
- `AppState`: Manages all application state (orientation, waypoints, parameters)

### Widgets & Sections
- `TitleBar`: Application title
- `ResetButton`: Reset all to defaults
- `TrackOrientationSection`: Track layout selector
- `StartPointSection`: Starting column selector
- `ManualWaypointSection`: Waypoint input controls
- `WaypointsDisplay`: Current waypoints table
- `ParametersSection`: Time parameter inputs
- `SummarySection`: Calculations and metrics
- `GeneratedCodeSection`: Rust-like code output

### Helper Functions
- `_rotate_vector()`: Rotate vectors for orientation calculations
- `_to_roman()`: Convert numbers to Roman numerals

## Calculation Details

The app performs complex calculations including:

1. **Distance Calculation**: Euclidean distance between waypoints
2. **Angle Calculation**: Rotation angles between consecutive waypoints
3. **Turn Classification**: Straight (S), Left (L), or Right (R) turns
4. **Time Allocation**: 
   - Angular time based on turn degrees
   - Linear time based on distance (Roman numeral classification)
   - Total time = sum of angular + linear times per waypoint

## Differences from Streamlit Version

- **Layout**: Two-column layout (left: main controls, right: parameters)
- **Interaction**: Keyboard and mouse-driven TUI vs. web interface
- **Performance**: Faster, no need for browser or web server
- **Portability**: Runs in any terminal supporting ANSI colors

## System Requirements

- Python 3.8+
- macOS, Linux, or Windows with terminal support
- Textual 0.1.0+ (tested with 8.2.3)

## Verification

All functionality has been tested and verified to work correctly:
- ✓ Module imports successful
- ✓ Core calculations (distance, angles, Roman numerals)
- ✓ State management and reset functionality
- ✓ Track orientation handling (4x5 and 5x4)
- ✓ Waypoint type management (Forward, Backward, Gate)
- ✓ Parameter handling and defaults
- ✓ UI component instantiation

## Future Enhancements

Potential improvements:
- Visual grid display in the TUI
- Edit waypoint functionality in UI
- Mouse support for waypoint selection
- Color-coded waypoint types
- Export functionality (JSON, CSV)
- Scrollable code output
- Real-time calculations update
