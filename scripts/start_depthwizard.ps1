# DepthWizard (SIH26175) — System Launcher
# ISRO / Department of Space — Smart India Hackathon 2026

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item $ScriptDir).Parent.FullName
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $PythonExe) {
    & $PythonExe (Join-Path $ScriptDir "start_depthwizard.py")
} else {
    python (Join-Path $ScriptDir "start_depthwizard.py")
}
