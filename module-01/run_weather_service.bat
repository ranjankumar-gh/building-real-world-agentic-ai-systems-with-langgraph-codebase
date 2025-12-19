@echo off
echo ========================================
echo Starting Local Weather API Service
echo ========================================
echo.
echo The service will be available at:
echo - API: http://localhost:8000
echo - Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo ========================================
echo.

python weather_api.py
