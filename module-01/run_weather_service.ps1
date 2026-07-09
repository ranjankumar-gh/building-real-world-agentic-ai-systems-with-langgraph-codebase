Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Local Weather API Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The service will be available at:"
Write-Host " - API:  http://localhost:8000" -ForegroundColor Green
Write-Host " - Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python weather_api.py