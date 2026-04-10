# Robot Tour - Setup & Installation

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install streamlit>=1.28.0
```

### 2. Run the Application

**superSimple.py (Fast, text-based):**
```bash
streamlit run superSimple.py
```


## Package Details

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.28.0 | Web-based UI framework |

## Troubleshooting

**"streamlit: command not found"**
- Ensure streamlit is installed: `pip install streamlit`
- Check pip is pointing to correct Python: `pip --version`

**Module not found errors**
- Reinstall: `pip install --upgrade -r requirements.txt`

**Port already in use**
- Streamlit runs on port 8501. If busy, use: `streamlit run file.py --server.port 8502`
