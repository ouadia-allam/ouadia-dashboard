import os
import subprocess
import sys

def launch_dashboard():
    """Launch the Financial Dashboard application."""
    print("Starting Financial Dashboard...")
    
    # Path to the streamlit script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Command to run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", "5000"]
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_dashboard()