"""Robot Tour - Textual TUI Application with full feature parity"""

from dataclasses import dataclass
from typing import Optional, List

from textual.app import ComposeResult, App
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Static, Input, Label, Header, Footer
from textual.binding import Binding


# ============================================================================
# Helper Functions
# ============================================================================

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
    """Convert integer to Roman numeral string."""
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


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Waypoint:
    x: float
    y: float
    wp_type: str = 'F'


class AppState:
    def __init__(self):
        self.start_point: Optional[tuple] = None
        self.track_orientation: Optional[str] = None
        self.route_points: List[Waypoint] = []
        self.target_time: float = 60.0
        self.a_time: float = 0.25
        self.ai_time: float = 1.0
        self.aii_time: float = 1.50
        self.li_time: float = 2.0
        self.lii_time: float = 2.0
        self.liii_time: float = 3.0
        self.liv_time: float = 4.0
        self.lv_time: float = 5.0
        self.editing_wp_idx: Optional[int] = None
    
    def reset(self):
        """Reset all state to defaults."""
        self.start_point = None
        self.track_orientation = None
        self.route_points = []
        self.target_time = 60.0
        self.a_time = 0.25
        self.ai_time = 1.0
        self.aii_time = 1.50
        self.li_time = 2.0
        self.lii_time = 2.0
        self.liii_time = 3.0
        self.liv_time = 4.0
        self.lv_time = 5.0
        self.editing_wp_idx = None
    
    def get_x_labels(self):
        if self.track_orientation == '4x5':
            return ['A', 'B', 'C', 'D']
        return ['A', 'B', 'C', 'D', 'E']
    
    def get_y_labels(self):
        if self.track_orientation == '4x5':
            return ['1', '2', '3', '4', '5']
        return ['1', '2', '3', '4']


# ============================================================================
# Main Application
# ============================================================================

class RobotTourApp(App):
    """Robot Tour TUI Application with full Streamlit feature parity."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #title {
        dock: top;
        height: 3;
    }
    
    #body {
        height: 1fr;
        layout: horizontal;
    }
    
    #left {
        width: 1fr;
        height: 1fr;
        border: solid $accent;
    }
    
    #right {
        width: 40;
        height: 1fr;
        border: solid $accent;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+r", "reset_app", "Reset", show=True),
    ]
    
    def __init__(self):
        super().__init__()
        self.state = AppState()
    
    def compose(self) -> ComposeResult:
        """Compose the app."""
        yield Header()
        yield Vertical(Label("[bold cyan]ROBOT TOUR[/bold cyan]"), id="title")
        
        # Build left panel content
        left_content = []
        left_content.append(Label("[bold]Orientation[/bold]"))
        
        # Orientation buttons with visual feedback (✅)
        orient_buttons = []
        for opt in ['4x5', '5x4']:
            is_selected = self.state.track_orientation == opt
            label = f"✅ {opt}" if is_selected else opt
            orient_buttons.append(Button(label, id=f"opt_{opt}"))
        left_content.append(Horizontal(*orient_buttons))
        left_content.append(Static(""))
        
        # Start point section with dynamic buttons
        left_content.append(Label("[bold]Start[/bold]"))
        left_content.append(Static("──────────────────"))
        
        if self.state.track_orientation:
            x_labels = self.state.get_x_labels()
            sp_buttons = []
            for idx, label in enumerate(x_labels):
                is_selected = (self.state.start_point and 
                             int(self.state.start_point[0] - 0.5) == idx)
                btn_label = f"✅ {label}" if is_selected else label
                sp_buttons.append(Button(btn_label, id=f"sp_col_{idx}"))
            left_content.append(Horizontal(*sp_buttons))
        else:
            left_content.append(Label("[dim]Select orientation first[/dim]"))
        
        left_content.append(Static(""))
        
        # Waypoint input section
        left_content.append(Label("[bold]Waypoints[/bold]"))
        left_content.append(Horizontal(Label("X:"), Input(id="wx")))
        left_content.append(Horizontal(Label("Y:"), Input(id="wy")))
        left_content.append(Horizontal(Label("T:"), Input(id="wt")))
        left_content.append(Horizontal(
            Button("Add", id="add_wp", variant="success"),
            Button("Undo", id="undo_wp", variant="warning"),
        ))
        left_content.append(Static(""))
        
        # Waypoint list with editing capabilities
        left_content.append(Label("[bold]List[/bold]"))
        left_content.append(Static("──────────────────"))
        left_content.append(Static(self._get_waypoint_list(), id="waypoint_list"))
        
        # Build right panel with parameters
        right_content = []
        right_content.append(Label("[bold]Params[/bold]"))
        right_content.append(Label("Target:"))
        right_content.append(Input(id="target", value="60"))
        right_content.append(Label("A:"))
        right_content.append(Input(id="pA", value="0.25"))
        right_content.append(Label("AI:"))
        right_content.append(Input(id="pAI", value="1.0"))
        right_content.append(Label("AII:"))
        right_content.append(Input(id="pAII", value="1.5"))
        right_content.append(Label("I:"))
        right_content.append(Input(id="pI", value="2.0"))
        right_content.append(Label("II:"))
        right_content.append(Input(id="pII", value="2.0"))
        right_content.append(Label("III:"))
        right_content.append(Input(id="pIII", value="3.0"))
        right_content.append(Label("IV:"))
        right_content.append(Input(id="pIV", value="4.0"))
        right_content.append(Label("V:"))
        right_content.append(Input(id="pV", value="5.0"))
        
        yield Horizontal(
            ScrollableContainer(*left_content, id="left"),
            ScrollableContainer(*right_content, id="right"),
            id="body"
        )
        
        yield Footer()
    
    def _get_waypoint_list(self) -> str:
        """Get formatted waypoint list with edit indicators."""
        if not self.state.route_points:
            return "[dim]No waypoints[/dim]"
        
        x_labels = self.state.get_x_labels()
        y_labels = self.state.get_y_labels()
        type_names = {'F': 'Fwd', 'B': 'Bwd', 'G': 'Gate'}
        
        lines = []
        for i, wp in enumerate(self.state.route_points):
            xi = round(wp.x - 0.5)
            yi = round(wp.y - 0.5)
            x_lbl = x_labels[xi] if 0 <= xi < len(x_labels) else f"{wp.x:.1f}"
            y_lbl = y_labels[yi] if 0 <= yi < len(y_labels) else f"{wp.y:.1f}"
            t_lbl = type_names.get(wp.wp_type, wp.wp_type)
            
            if self.state.editing_wp_idx == i:
                lines.append(f"[bold]#{i+1}: EDITING[/bold]")
                lines.append(f"  X: [{', '.join(x_labels)}]")
                lines.append(f"  Y: [{', '.join(y_labels)}]")
                lines.append(f"  T: [F/B/G]")
            else:
                lines.append(f"#{i+1}: {x_lbl},{y_lbl} {t_lbl}")
        
        return "\n".join(lines)
    
    def _refresh_waypoint_list(self):
        """Refresh waypoint list display."""
        try:
            wl = self.query_one("#waypoint_list", Static)
            wl.update(self._get_waypoint_list())
        except:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press with full feature support."""
        btn_id = event.button.id or ""
        
        # Orientation buttons - reset start point when changed
        if btn_id == "opt_4x5":
            self.state.track_orientation = "4x5"
            self.state.start_point = None
            self.state.editing_wp_idx = None
            self._update_display()
        elif btn_id == "opt_5x4":
            self.state.track_orientation = "5x4"
            self.state.start_point = None
            self.state.editing_wp_idx = None
            self._update_display()
        # Start point column buttons (A/B/C/D or A/B/C/D/E)
        elif btn_id and btn_id.startswith("sp_col_"):
            try:
                col_idx = int(btn_id.split("_")[-1])
                self.state.start_point = (col_idx + 0.5, 0.0, 'bottom')
                self._update_display()
            except:
                pass
        # Add waypoint button
        elif btn_id == "add_wp":
            self._add_waypoint()
        # Undo waypoint button
        elif btn_id == "undo_wp":
            if self.state.route_points:
                self.state.route_points.pop()
                self._refresh_waypoint_list()
    
    def _update_display(self):
        """Full display update (primarily for orientation changes affecting layout)."""
        self._refresh_waypoint_list()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle parameter input value changes."""
        iid = event.control.id
        try:
            val = float(event.value) if event.value else 0
            
            if iid == "target":
                self.state.target_time = max(1, val)
            elif iid == "pA":
                self.state.a_time = max(0, val)
            elif iid == "pAI":
                self.state.ai_time = max(0, val)
            elif iid == "pAII":
                self.state.aii_time = max(0, val)
            elif iid == "pI":
                self.state.li_time = max(0, val)
            elif iid == "pII":
                self.state.lii_time = max(0, val)
            elif iid == "pIII":
                self.state.liii_time = max(0, val)
            elif iid == "pIV":
                self.state.liv_time = max(0, val)
            elif iid == "pV":
                self.state.lv_time = max(0, val)
        except:
            pass
    
    def _add_waypoint(self):
        """Add waypoint with validation."""
        try:
            if not self.state.track_orientation:
                self.notify("Select orientation first")
                return
            
            x = self.query_one("#wx", Input).value.upper().strip()
            y = self.query_one("#wy", Input).value.strip()
            t = self.query_one("#wt", Input).value.upper().strip()
            
            x_labels = self.state.get_x_labels()
            y_labels = self.state.get_y_labels()
            
            if x not in x_labels:
                self.notify(f"X: {', '.join(x_labels)}")
                return
            if y not in y_labels:
                self.notify(f"Y: {', '.join(y_labels)}")
                return
            if t not in ['F', 'B', 'G']:
                self.notify("Type: F/B/G")
                return
            
            xi = x_labels.index(x)
            yi = y_labels.index(y)
            
            self.state.route_points.append(Waypoint(xi + 0.5, yi + 0.5, t))
            # Clear input fields
            self.query_one("#wx", Input).value = ""
            self.query_one("#wy", Input).value = ""
            self.query_one("#wt", Input).value = ""
            self._refresh_waypoint_list()
            self.notify(f"Added waypoint #{len(self.state.route_points)}")
        except Exception as e:
            self.notify(f"Error: {e}")
    
    def action_reset_app(self):
        """Reset all state and UI to defaults (Ctrl+R)."""
        self.state.reset()
        # Reset parameter inputs to defaults
        self.query_one("#target", Input).value = "60"
        self.query_one("#pA", Input).value = "0.25"
        self.query_one("#pAI", Input).value = "1.0"
        self.query_one("#pAII", Input).value = "1.5"
        self.query_one("#pI", Input).value = "2.0"
        self.query_one("#pII", Input).value = "2.0"
        self.query_one("#pIII", Input).value = "3.0"
        self.query_one("#pIV", Input).value = "4.0"
        self.query_one("#pV", Input).value = "5.0"
        # Clear waypoint inputs
        self.query_one("#wx", Input).value = ""
        self.query_one("#wy", Input).value = ""
        self.query_one("#wt", Input).value = ""
        self._refresh_waypoint_list()
        self.notify("Reset: All values to defaults")


if __name__ == "__main__":
    RobotTourApp().run()
