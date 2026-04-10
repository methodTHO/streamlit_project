# Feature Comparison: Streamlit vs Textual

## Summary
The Textual app is **partially complete** with some critical features missing that prevent full feature parity with the Streamlit version.

---

## Feature Checklist

### 1. **Reset Button** ✅ COMPLETE
- **Streamlit**: Right-aligned to "Robot Tour" title, resets ALL values to defaults
- **Textual**: Present and functional ✅
- **Status**: WORKING

### 2. **Track Orientation Selector** ⚠️ PARTIAL
- **Streamlit**: Two buttons (4x5, 5x4) with ✅ visual feedback when selected
- **Textual**: Two buttons present BUT missing ✅ feedback
- **Status**: NEEDS VISUAL FEEDBACK

### 3. **Start Point Selection** ❌ MISSING ⚠️ CRITICAL
- **Streamlit**: Dynamic column buttons showing ✅ when selected
  - 4x5 orientation: 4 buttons (A, B, C, D)
  - 5x4 orientation: 5 buttons (A, B, C, D, E)
  - Resets when orientation changes
  - Location: Lines 154-171
- **Textual**: Only shows label "Start", NO interactive buttons
- **Status**: NOT IMPLEMENTED - **BLOCKS USER FROM SETTING START POINT**

### 4. **Manual Waypoint Input** ⚠️ DIFFERENT
- **Streamlit**: 3 selectboxes (X, Y, Type) + Add/Undo buttons
  - X options: A/B/C/D or A/B/C/D/E (based on orientation)
  - Y options: 1/2/3/4/5 or 1/2/3/4 (based on orientation)
  - Type options: Forward, Backward, Gate
  - Shows hint text with allowed values
- **Textual**: 3 text inputs (wx, wy, wt) + Add/Undo buttons
  - Free text entry (less user-friendly but functional)
  - Does validate input against allowed values
  - Shows error notifications for invalid input
- **Status**: FUNCTIONAL BUT LESS USER-FRIENDLY (text vs dropdowns)

### 5. **Waypoint Display & Editing** ❌ MISSING
- **Streamlit**: Shows table list of all waypoints with:
  - Display mode: "1. A,2 — Forward" with Edit button
  - Edit mode: Selectboxes for X/Y/Type with Save/Cancel buttons
  - Location: Lines 223-254
- **Textual**: Only shows simple info display (read-only)
  - Shows "WP: N" (waypoint count) but not individual waypoints
  - NO edit functionality
  - NO delete/save/cancel buttons
- **Status**: NOT IMPLEMENTED - Blocks waypoint editing

### 6. **Parameter Inputs** ✅ COMPLETE
All 9 parameters present in Textual:
- ✅ Target time
- ✅ A, AI, AII (Angular times)
- ✅ I, II, III, IV, V (Linear times)
- **Status**: WORKING

### 7. **Default Values** ✅ COMPLETE
All defaults match Streamlit:
- ✅ target_time: 60s
- ✅ a_time: 0.25
- ✅ ai_time: 1.0
- ✅ aii_time: 1.50
- ✅ li_time: 2.0
- ✅ lii_time: 2.0
- ✅ liii_time: 3.0
- ✅ liv_time: 4.0
- ✅ lv_time: 5.0
- **Status**: WORKING

---

## Critical Issues

### 🔴 BLOCKING ISSUE: No Start Point Buttons
**Impact**: User cannot set start point at all
**Streamlit Location**: Lines 154-171
**TUI Location**: Line 158-160 (currently just a label)
**Solution**: Dynamically generate column buttons (A/B/C/D or A/B/C/D/E) based on orientation

### 🔴 HIGH PRIORITY: Waypoint Editing Missing
**Impact**: Users cannot edit or review waypoints in detail
**Streamlit Location**: Lines 223-254
**TUI Location**: No current implementation
**Solution**: Add waypoint list display with Edit button, implement save/cancel flow

### 🟡 MEDIUM PRIORITY: Orientation Visual Feedback
**Impact**: Users can't see which orientation is selected
**Streamlit**: Shows ✅ indicator on selected button
**TUI**: Buttons plain, no feedback
**Solution**: Add ✅ indicator or color change to selected orientation button

---

## Code Locations Reference

### Streamlit (superSimple.py):
- Reset button: Lines 115-135
- Orientation selector: Lines 140-152
- **Start Point selector: Lines 154-171** ← MISSING IN TUI
- Manual waypoint input: Lines 174-220
- **Waypoint display & editing: Lines 223-254** ← MISSING IN TUI
- Parameters: Lines 257+

### Textual (robot_tour_tui.py):
- AppState defaults: Lines 53-85 ✅
- Orientation buttons: Lines 146-156 (needs ✅ feedback)
- **Start Point: Lines 158-160** (currently just label, needs buttons)
- **Waypoint inputs: Lines 161-170** (text-based, not selectboxes)
- **Info display: Lines 172-183** (read-only, needs edit functionality)
- Parameter inputs: Lines 185-193 ✅

---

## User-Facing Functionality Loss

| Feature | Streamlit | Textual | Impact |
|---------|-----------|---------|--------|
| Start Point Selection | ✅ Interactive buttons | ❌ Label only | **CRITICAL** - Can't set start |
| Waypoint Editing | ✅ Full edit UI | ❌ Missing | **HIGH** - Can't edit added waypoints |
| Orientation Feedback | ✅ ✅ indicator | ❌ Plain buttons | MINOR - Can't see selection |
| Waypoint Review | ✅ Table list | ❌ Info only | MEDIUM - Can't see waypoint details |
| Input Method | ✅ Dropdowns | ⚠️ Text input | LOW - Different but works |

---

## Recommendation

**The Textual app is functionally incomplete and needs these changes to achieve feature parity:**

1. **Immediate (Required for basic functionality)**: Add start point column buttons
2. **Soon (Required for full usability)**: Add waypoint list with edit functionality  
3. **Nice-to-have**: Add ✅ feedback to orientation buttons + improve waypoint input UI

The app currently missing ~30% of the UI features that make the Streamlit version fully functional.
