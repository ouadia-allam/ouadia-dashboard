Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = oWS.SpecialFolders("Desktop") & "\Financial Dashboard.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)

' Get the current script path
strPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Check if we're in development or distribution mode
' In distribution mode, the script is in the same folder as the .exe
' In development mode, the .exe is in the dist subfolder
If CreateObject("Scripting.FileSystemObject").FileExists(strPath & "\FinancialDashboard.exe") Then
    ' We're in distribution mode (running from dist folder)
    targetExe = strPath & "\FinancialDashboard.exe"
    workingDir = strPath
    iconPath = strPath & "\generated-icon.png, 0"
Else
    ' We're in development mode
    targetExe = strPath & "\dist\FinancialDashboard.exe"
    workingDir = strPath
    iconPath = strPath & "\generated-icon.png, 0"
End If

' Set shortcut properties
oLink.TargetPath = targetExe
oLink.WorkingDirectory = workingDir
oLink.Description = "Financial Dashboard Application"

' Set icon if exists
If CreateObject("Scripting.FileSystemObject").FileExists(Split(iconPath, ",")(0)) Then
    oLink.IconLocation = iconPath
End If

oLink.Save

' Show a message
MsgBox "Financial Dashboard shortcut created on your desktop!", 64, "Shortcut Created"
