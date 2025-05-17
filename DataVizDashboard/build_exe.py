import os
import sys
import subprocess
import platform
import shutil

def build_executable():
    """Build the Financial Dashboard executable."""
    print("Starting build process for Financial Dashboard...")
    
    # Get the current directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Define the paths
    icon_path = os.path.join(base_dir, 'generated-icon.png')
    
    # Build the PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=FinancialDashboard",
        "--onefile",
        f"--add-data={os.path.join(base_dir, 'data_service.py')}{';' if platform.system() == 'Windows' else ':'}.",
        f"--add-data={os.path.join(base_dir, 'utils.py')}{';' if platform.system() == 'Windows' else ':'}.",
        f"--add-data={os.path.join(base_dir, 'enhanced_economic_tab.py')}{';' if platform.system() == 'Windows' else ':'}.",
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=plotly",
        "--hidden-import=plotly.graph_objects",
        "--hidden-import=plotly.express",
        "--hidden-import=plotly.subplots",
        "--collect-data=streamlit",
        "--collect-data=plotly",
        "--noconfirm",
        "--clean",
    ]
    
    # Add icon if it exists
    if os.path.exists(icon_path):
        cmd.append(f"--icon={icon_path}")
    
    # Add the main script
    cmd.append("app.py")
    
    # Run PyInstaller
    print("Running PyInstaller with the following command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if build was successful
    if result.returncode == 0:
        print("\n✅ Build successful! The executable has been created.")
        print(f"\nExecutable location: {os.path.join(base_dir, 'dist', 'FinancialDashboard.exe')}")
        
        # Copy shortcut creation files to dist folder
        print("\nCopying shortcut creation files to dist folder...")
        dist_dir = os.path.join(base_dir, 'dist')
        
        # Files to copy
        shortcut_files = [
            'create_desktop_shortcut.vbs',
            'CreateShortcut.bat',
            'FinancialDashboardSetup.bat',
            'generated-icon.png'
        ]
        
        for file in shortcut_files:
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                shutil.copy2(file_path, dist_dir)
                print(f"  ✓ Copied {file} to dist folder")
            else:
                print(f"  ⚠️ Warning: {file} not found, skipping")
        
        print("\nCreating distribution package...")
        # Create a README file in the dist folder
        readme_content = """Financial Dashboard - Standalone Application

Quick Start Guide:
-----------------

1. Run the Application:
   Double-click FinancialDashboard.exe to start the application

2. Create Desktop Shortcut (3 options):
   
   Option A: Quickest Setup
   - Double-click FinancialDashboardSetup.bat for one-click setup
   
   Option B: Manual Setup
   - Double-click CreateShortcut.bat to create a desktop shortcut
   
   Option C: Traditional Way
   - Right-click on FinancialDashboard.exe
   - Select "Create shortcut"
   - Move the shortcut to your desktop

Notes:
- No installation or Python required!
- This is a completely standalone application
- All necessary components are included in the .exe file"""
        
        with open(os.path.join(dist_dir, 'README.txt'), 'w') as f:
            f.write(readme_content)
        
        print("  ✓ Created README.txt file")
        
        # Suggest creating a zip file of the dist folder for easy distribution
        print("\nTo distribute the application:")
        print("1. Zip the entire 'dist' folder")
        print("2. Share the zip file with users")
        print("3. Users can simply extract and run FinancialDashboard.exe")
        print("4. For desktop shortcut, users can run CreateShortcut.bat")
    else:
        print("\n❌ Build failed. Error details:")
        print(result.stderr)
        print("\nTry running the build manually with:")
        print(f"cd {base_dir} && pyinstaller --onefile --name=FinancialDashboard app.py")

if __name__ == "__main__":
    build_executable()