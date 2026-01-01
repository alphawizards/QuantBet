# Task Scheduler Setup Script (PowerShell)
# Automatically creates Windows Task Scheduler job for daily monitoring

param(
    [string]$TaskTime = "08:00",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "QUANTBET TASK SCHEDULER SETUP" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`n❌ ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✅ Running as Administrator" -ForegroundColor Green

# Configuration
$TaskName = "QuantBet Daily Monitoring"
$ScriptPath = Join-Path $PSScriptRoot "daily_monitoring.ps1"
$WorkingDir = Split-Path -Parent $PSScriptRoot

# Verify script exists
if (!(Test-Path $ScriptPath)) {
    Write-Host "`n❌ ERROR: Monitoring script not found at: $ScriptPath" -ForegroundColor Red
    exit 1
}

Write-Host "`nConfiguration:" -ForegroundColor Cyan
Write-Host "  Task Name: $TaskName"
Write-Host "  Script: $ScriptPath"
Write-Host "  Working Dir: $WorkingDir"
Write-Host "  Schedule: Daily at $TaskTime"

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existingTask -and !$Force) {
    Write-Host "`n⚠️  Task '$TaskName' already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to replace it? (y/n)"
    
    if ($response -ne 'y') {
        Write-Host "Setup cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Remove existing task if present
if ($existingTask) {
    Write-Host "`nRemoving existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "✅ Existing task removed" -ForegroundColor Green
}

# Create scheduled task action
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$ScriptPath`" -Verbose" `
    -WorkingDirectory $WorkingDir

# Create trigger (daily at specified time)
$trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $TaskTime

# Create task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# Create task principal (run whether user is logged in or not)
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel Limited

# Register the task
Write-Host "`nCreating scheduled task..." -ForegroundColor Cyan

try {
    $task = Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Daily monitoring of QuantBet model performance and system health"
    
    Write-Host "✅ Task created successfully!" -ForegroundColor Green
    
}
catch {
    Write-Host "❌ Failed to create task: $_" -ForegroundColor Red
    exit 1
}

# Test run the task
Write-Host "`nWould you like to test run the task now? (y/n): " -ForegroundColor Cyan -NoNewline
$testRun = Read-Host

if ($testRun -eq 'y') {
    Write-Host "`nStarting test run..." -ForegroundColor Yellow
    
    try {
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "✅ Task started! Check the output in:" -ForegroundColor Green
        Write-Host "  $WorkingDir\data\monitoring\daily_monitor_$(Get-Date -Format 'yyyyMMdd').log" -ForegroundColor Cyan
        
        # Wait a bit for task to start
        Start-Sleep -Seconds 2
        
        # Check task status
        $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Host "`nTask Status: $($taskInfo.LastTaskResult)" -ForegroundColor Cyan
        Write-Host "Last Run Time: $($taskInfo.LastRunTime)" -ForegroundColor Cyan
        
    }
    catch {
        Write-Host "⚠️  Test run failed: $_" -ForegroundColor Yellow
    }
}

# Summary
Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host "`nTask Details:"
Write-Host "  Name: $TaskName"
Write-Host "  Schedule: Daily at $TaskTime"
Write-Host "  Next Run: $((Get-ScheduledTask -TaskName $TaskName).Triggers[0].StartBoundary)"

Write-Host "`nManagement Commands:"
Write-Host "  View task:    Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
Write-Host "  Run now:      Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
Write-Host "  Disable:      Disable-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
Write-Host "  Remove:       Unregister-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan

Write-Host "`nLogs Location:"
Write-Host "  $WorkingDir\data\monitoring\" -ForegroundColor Yellow

Write-Host "`n✅ Daily monitoring is now automated!" -ForegroundColor Green
Write-Host ""
