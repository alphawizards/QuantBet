# QuantBet Daily Monitoring Script (PowerShell)
# Runs model monitoring dashboard and sends alerts if needed

param(
    [int]$Days = 7,
    [string]$AlertEmail = "",
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Configuration
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$PYTHON_CMD = "python"
$LOG_FILE = Join-Path $PROJECT_ROOT "data\monitoring\daily_monitor_$(Get-Date -Format 'yyyyMMdd').log"

# Create log directory
$LOG_DIR = Split-Path -Parent $LOG_FILE
if (!(Test-Path -Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    Write-Output $logMessage | Tee-Object -FilePath $LOG_FILE -Append
    
    if ($Verbose) {
        switch ($Level) {
            "ERROR" { Write-Host $logMessage -ForegroundColor Red }
            "WARNING" { Write-Host $logMessage -ForegroundColor Yellow }
            "SUCCESS" { Write-Host $logMessage -ForegroundColor Green }
            default { Write-Host $logMessage }
        }
    }
}

Write-Log "=" * 60
Write-Log "QUANTBET DAILY MONITORING" "INFO"
Write-Log "=" * 60

# Step 1: Check if Docker containers are running
Write-Log "Checking Docker containers..." "INFO"

try {
    $containers = docker ps --format "{{.Names}}" 2>&1
    
    if ($containers -match "quantbet_api") {
        Write-Log "✅ API container is running" "SUCCESS"
    }
    else {
        Write-Log "⚠️  API container not running" "WARNING"
    }
    
    if ($containers -match "quantbet_db") {
        Write-Log "✅ Database container is running" "SUCCESS"
    }
    else {
        Write-Log "❌ Database container not running" "ERROR"
        exit 1
    }
}
catch {
    Write-Log "❌ Failed to check Docker status: $_" "ERROR"
}

# Step 2: Run model monitoring dashboard
Write-Log "`nRunning model monitoring dashboard..." "INFO"

try {
    Push-Location $PROJECT_ROOT
    
    $monitorOutput = & $PYTHON_CMD -m src.monitoring.dashboard 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Log "✅ Monitoring dashboard executed successfully" "SUCCESS"
        
        # Parse output for key metrics
        $output_str = $monitorOutput | Out-String
        
        if ($output_str -match "Brier Score: ([\d.]+)") {
            $brierScore = [float]$matches[1]
            Write-Log "  Brier Score: $brierScore" "INFO"
            
            if ($brierScore -gt 0.25) {
                Write-Log "  ⚠️  HIGH BRIER SCORE DETECTED" "WARNING"
            }
        }
        
        if ($output_str -match "Accuracy: ([\d.]+)%") {
            $accuracy = [float]$matches[1]
            Write-Log "  Accuracy: $accuracy%" "INFO"
            
            if ($accuracy -lt 48) {
                Write-Log "  ⚠️  LOW ACCURACY DETECTED" "WARNING"
            }
        }
        
        if ($output_str -match "ROI: ([-\d.]+)%") {
            $roi = [float]$matches[1]
            Write-Log "  ROI: $roi%" "INFO"
            
            if ($roi -lt -5) {
                Write-Log "  🚨 NEGATIVE ROI ALERT" "ERROR"
            }
        }
        
    }
    else {
        Write-Log "⚠️  Monitoring dashboard returned exit code: $exitCode" "WARNING"
    }
    
    Pop-Location
    
}
catch {
    Write-Log "❌ Failed to run monitoring dashboard: $_" "ERROR"
}

# Step 3: Check prediction logs
Write-Log "`nChecking prediction logs..." "INFO"

$predictionDir = Join-Path $PROJECT_ROOT "data\predictions"

if (Test-Path -Path $predictionDir) {
    $recentLogs = Get-ChildItem -Path $predictionDir -Filter "*.jsonl" | 
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-$Days) } |
    Sort-Object LastWriteTime -Descending
    
    if ($recentLogs.Count -gt 0) {
        Write-Log "✅ Found $($recentLogs.Count) prediction log files from last $Days days" "SUCCESS"
        
        # Count total predictions
        $totalPredictions = 0
        foreach ($log in $recentLogs) {
            $lines = Get-Content $log.FullName | Measure-Object -Line
            $totalPredictions += $lines.Lines
        }
        
        Write-Log "  Total predictions logged: $totalPredictions" "INFO"
        
        if ($totalPredictions -lt 10) {
            Write-Log "  ⚠️  Low prediction volume" "WARNING"
        }
        
    }
    else {
        Write-Log "  ⚠️  No prediction logs found in last $Days days" "WARNING"
    }
}
else {
    Write-Log "  ⚠️  Prediction log directory not found" "WARNING"
}

# Step 4: Generate summary report
Write-Log "`nGenerating summary report..." "INFO"

$reportPath = Join-Path $PROJECT_ROOT "data\monitoring\daily_report_$(Get-Date -Format 'yyyyMMdd').txt"

try {
    @"
QUANTBET DAILY MONITORING REPORT
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Period: Last $Days days

SYSTEM STATUS:
- Docker Containers: $(if ($containers -match "quantbet") { "✅ Running" } else { "❌ Stopped" })
- API Health: $(if ($containers -match "quantbet_api") { "✅ Healthy" } else { "⚠️  Check Required" })
- Database: $(if ($containers -match "quantbet_db") { "✅ Connected" } else { "❌ Down" })

MODEL METRICS:
$(if ($brierScore) { "- Brier Score: $brierScore" } else { "- Brier Score: N/A" })
$(if ($accuracy) { "- Accuracy: $accuracy%" } else { "- Accuracy: N/A" })
$(if ($roi) { "- ROI: $roi%" } else { "- ROI: N/A" })

PREDICTION ACTIVITY:
$(if ($totalPredictions) { "- Total Predictions: $totalPredictions" } else { "- Total Predictions: 0" })
$(if ($recentLogs) { "- Log Files: $($recentLogs.Count)" } else { "- Log Files: 0" })

RECOMMENDATIONS:
$(if ($brierScore -and $brierScore -gt 0.25) { "⚠️  Consider model retraining - high Brier score`n" } else { "" })
$(if ($accuracy -and $accuracy -lt 48) { "⚠️  Review model calibration - low accuracy`n" } else { "" })
$(if ($roi -and $roi -lt -5) { "🚨 URGENT: Negative ROI - review bet sizing and edge thresholds`n" } else { "" })
$(if ($totalPredictions -lt 10) { "⚠️  Low prediction volume - check data pipeline`n" } else { "" })
$(if (!$brierScore -and !$accuracy -and !$roi) { "ℹ️  Insufficient data for metrics - wait for more resolved predictions`n" } else { "" })

NEXT ACTIONS:
1. Review full monitoring dashboard: python -m src.monitoring.dashboard
2. Check recent prediction logs in data/predictions/
3. Validate model performance using walk-forward validation
4. Update models if metrics degrade

---
Log file: $LOG_FILE
"@ | Out-File -FilePath $reportPath -Encoding UTF8

    Write-Log "✅ Report saved to: $reportPath" "SUCCESS"
    
}
catch {
    Write-Log "❌ Failed to generate report: $_" "ERROR"
}

# Step 5: Send alerts if configured
if ($AlertEmail -ne "") {
    Write-Log "`nChecking alert conditions..." "INFO"
    
    $alertNeeded = $false
    $alertMessage = "QuantBet Monitoring Alerts:`n`n"
    
    if ($brierScore -and $brierScore -gt 0.25) {
        $alertNeeded = $true
        $alertMessage += "- HIGH BRIER SCORE: $brierScore > 0.25`n"
    }
    
    if ($accuracy -and $accuracy -lt 48) {
        $alertNeeded = $true
        $alertMessage += "- LOW ACCURACY: $accuracy% < 48%`n"
    }
    
    if ($roi -and $roi -lt -5) {
        $alertNeeded = $true
        $alertMessage += "- NEGATIVE ROI: $roi% < -5%`n"
    }
    
    if ($alertNeeded) {
        Write-Log "🚨 Alerts triggered - email functionality not implemented" "WARNING"
        Write-Log "  Would send to: $AlertEmail" "INFO"
        # TODO: Implement email sending with Send-MailMessage or external service
    }
    else {
        Write-Log "✅ No alerts - all metrics within thresholds" "SUCCESS"
    }
}

Write-Log "`n" + "=" * 60
Write-Log "MONITORING COMPLETE" "INFO"
Write-Log "=" * 60

exit 0
