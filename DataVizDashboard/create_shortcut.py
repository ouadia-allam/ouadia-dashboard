import os
import sys
import platform
import subprocess

def create_shortcut():
    """Create a desktop shortcut for the Financial Dashboard."""
    print("Shortcut Creation Utility for Financial Dashboard")
    print("="*50)
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("⚠️  This shortcut creation utility is for Windows only.")
        print("On other operating systems, please create shortcuts manually.")
        return False
    
    try:
        # Get the current script directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Path to the executable or batch file
        exe_path = os.path.join(current_dir, "dist", "FinancialDashboard.exe")
        bat_path = os.path.join(current_dir, "FinancialDashboard.bat")
        
        # Determine which file to use for the shortcut
        if os.path.exists(exe_path):
            target_path = exe_path
            print(f"✓ Found executable: {target_path}")
        elif os.path.exists(bat_path):
            target_path = bat_path
            print(f"✓ Found batch file: {target_path}")
        else:
            print("❌ Neither the executable nor the batch file could be found.")
            print(f"  - Looked for: {exe_path}")
            print(f"  - Looked for: {bat_path}")
            return False
        
        # Icon path
        icon_path = os.path.join(current_dir, "generated-icon.png")
        
        # Get desktop path (works on Windows)
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        
        # Shortcut path
        shortcut_path = os.path.join(desktop, "Financial Dashboard.lnk")
        
        print("\nAttempting to create shortcut using PowerShell...")
        
        # Create PowerShell command for shortcut creation
        ps_command = f"""
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
        $Shortcut.TargetPath = '{target_path}'
        $Shortcut.WorkingDirectory = '{current_dir}'
        $Shortcut.Save()
        """
        
        # Run PowerShell command
        cmd = ['powershell', '-Command', ps_command]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Shortcut created successfully on the desktop: {shortcut_path}")
            return True
        else:
            print(f"❌ Error creating shortcut with PowerShell: {result.stderr}")
            print("\nAlternative method: Manual shortcut creation")
            print("1. Navigate to the 'dist' folder in this project")
            print("2. Right-click on FinancialDashboard.exe")
            print("3. Select 'Create shortcut'")
            print("4. Move the shortcut to your desktop")
            return False
            
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        print("\nPlease create a shortcut manually:")
        print("1. Navigate to the 'dist' folder in this project")
        print("2. Right-click on FinancialDashboard.exe")
        print("3. Select 'Create shortcut'")
        print("4. Move the shortcut to your desktop")
        return False

if __name__ == "__main__":
    success = create_shortcut()
    if not success:
        print("\nManual shortcut creation instructions:")
        print("1. Navigate to the following location:")
        print(f"   {os.path.abspath(os.path.join(os.path.dirname(__file__), 'dist'))}")
        print("2. Right-click on FinancialDashboard.exe")
        print("3. Select 'Create shortcut'")
        print("4. Move the shortcut to your desktop")
    
    input("\nPress Enter to exit...")