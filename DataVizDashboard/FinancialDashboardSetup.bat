@echo off
title Financial Dashboard Setup
echo ===================================
echo  Financial Dashboard Setup Utility
echo ===================================
echo.

echo Setting up Financial Dashboard...
echo.

:: Check if running from dist folder
if not exist "FinancialDashboard.exe" (
  echo ERROR: This setup must be run from the same folder as FinancialDashboard.exe
  echo.
  echo Please navigate to the folder containing FinancialDashboard.exe and run this setup again.
  goto END
)

:: Create desktop shortcut
echo Creating desktop shortcut...
if exist "create_desktop_shortcut.vbs" (
  cscript //nologo create_desktop_shortcut.vbs
) else (
  echo WARNING: Shortcut creation script not found.
  echo Creating shortcut manually...
  
  :: Create direct shortcut with pure batch
  echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\temp_shortcut.vbs"
  echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\Financial Dashboard.lnk" >> "%TEMP%\temp_shortcut.vbs"
  echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\temp_shortcut.vbs"
  echo oLink.TargetPath = "%CD%\FinancialDashboard.exe" >> "%TEMP%\temp_shortcut.vbs"
  echo oLink.WorkingDirectory = "%CD%" >> "%TEMP%\temp_shortcut.vbs"
  echo oLink.Description = "Financial Dashboard Application" >> "%TEMP%\temp_shortcut.vbs"
  echo oLink.Save >> "%TEMP%\temp_shortcut.vbs"
  
  cscript //nologo "%TEMP%\temp_shortcut.vbs"
  del "%TEMP%\temp_shortcut.vbs"
)

echo.
echo ===================================
echo  Setup Complete!
echo ===================================
echo.
echo Financial Dashboard has been set up on your computer.
echo.
echo You can now:
echo  1. Run FinancialDashboard.exe directly
echo  2. Use the desktop shortcut
echo.

:END
pause
