@echo off
title NeuroStage - Run Docker Cluster
cd /d "%~dp0"

echo ============================================================
echo  Starting NeuroStage Docker Services...
echo ============================================================
echo.

docker compose up -d

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo  ERROR: Docker failed to start.
    echo  If Docker Desktop is not running, right-click fix_docker.bat
    echo  and select 'Run as administrator'!
    echo ============================================================
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  NeuroStage is running successfully!
echo.
echo  - Frontend Web Dashboard: http://localhost:3000
echo  - Backend REST API Docs:  http://localhost:8000/docs
echo  - MLflow Tracking Server: http://localhost:5000
echo ============================================================
echo.
pause

