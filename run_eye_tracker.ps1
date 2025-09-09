# PowerShell script to run Eye Control Mouse on Windows
Write-Host "Starting Eye Control Mouse for Windows..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (Test-Path "eye_env\Scripts\activate") {
    Write-Host "Virtual environment found!" -ForegroundColor Yellow
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    
    # Activate virtual environment
    & "eye_env\Scripts\Activate.ps1"
    
    Write-Host "Starting the program..." -ForegroundColor Green
    python main.py
} else {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please make sure you have set up the virtual environment first." -ForegroundColor Red
    Write-Host "You can create it by running: python -m venv eye_env" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
