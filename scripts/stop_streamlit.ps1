# Stop Streamlit-related Python processes for ResearchAgent

Write-Host "Searching for Streamlit Python processes..." -ForegroundColor Cyan

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and
    $_.CommandLine -match "streamlit" -and
    $_.CommandLine -match "app.py"
}

if (-not $processes) {
    Write-Host "No Streamlit Python processes found." -ForegroundColor Green
    exit 0
}

foreach ($proc in $processes) {
    Write-Host "Stopping PID $($proc.ProcessId): $($proc.CommandLine)" -ForegroundColor Yellow
    Stop-Process -Id $proc.ProcessId -Force
}

Write-Host "Done." -ForegroundColor Green
