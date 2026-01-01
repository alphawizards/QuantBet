# Migration Test Script
# Description: Tests database migrations by applying and rolling back
# Usage: .\test_migrations.ps1 -Action [Apply|Rollback|Test]

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('Apply', 'Rollback', 'Test')]
    [string]$Action = 'Test',
    
    [Parameter(Mandatory = $false)]
    [string]$DatabaseName = 'quantbet',
    
    [Parameter(Mandatory = $false)]
    [string]$Host = 'localhost',
    
    [Parameter(Mandatory = $false)]
    [string]$User = 'postgres'
)

$ErrorActionPreference = 'Stop'
$MigrationsDir = [System.IO.Path]::Combine($PSScriptRoot, "..", "src", "database", "schema.sql", "migrations")

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "QuantBet Migration Test Script" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Yellow
Write-Host "Database: $DatabaseName" -ForegroundColor Yellow
Write-Host "Host: $Host" -ForegroundColor Yellow
Write-Host ""

# Function to execute SQL file
function Invoke-SqlFile {
    param(
        [string]$FilePath,
        [string]$Description
    )
    
    Write-Host "Executing: $Description..." -ForegroundColor Green
    Write-Host "  File: $(Split-Path -Leaf $FilePath)" -ForegroundColor Gray
    
    try {
        # Execute SQL file using psql
        $result = psql -h $Host -U $User -d $DatabaseName -f $FilePath 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED!" -ForegroundColor Red
            Write-Host "  Error: $result" -ForegroundColor Red
            throw "Migration failed: $Description"
        }
        
        Write-Host "  SUCCESS!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "  ERROR: $_" -ForegroundColor Red
        return $false
    }
}

# Function to verify table exists
function Test-TableExists {
    param([string]$TableName)
    
    $query = "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = '$TableName');"
    $result = psql -h $Host -U $User -d $DatabaseName -t -c $query
    
    return ($result -match 't')
}

# Function to apply all migrations
function Apply-Migrations {
    Write-Host "`nApplying migrations..." -ForegroundColor Cyan
    
    $migrations = @(
        @{File = "002_walk_forward_validation.sql"; Desc = "002: Walk-Forward Validation Tables" },
        @{File = "003_parameter_sensitivity.sql"; Desc = "003: Parameter Sensitivity Analysis" },
        @{File = "004_transaction_costs.sql"; Desc = "004: Transaction Cost Columns" },
        @{File = "005_monitoring_tables.sql"; Desc = "005: Production Monitoring Tables" }
    )
    
    foreach ($migration in $migrations) {
        $filePath = [System.IO.Path]::Combine($MigrationsDir, $migration.File)
        
        if (-not (Test-Path $filePath)) {
            Write-Host "  WARNING: Migration file not found: $($migration.File)" -ForegroundColor Yellow
            continue
        }
        
        $success = Invoke-SqlFile -FilePath $filePath -Description $migration.Desc
        if (-not $success) {
            Write-Host "`nMigration failed. Stopping." -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "`nAll migrations applied successfully!" -ForegroundColor Green
}

# Function to rollback all migrations
function Rollback-Migrations {
    Write-Host "`nRolling back migrations..." -ForegroundColor Cyan
    
    # Rollback in reverse order
    $rollbacks = @(
        @{File = "005_monitoring_tables_down.sql"; Desc = "005: Monitoring Tables (Rollback)" },
        @{File = "004_transaction_costs_down.sql"; Desc = "004: Transaction Costs (Rollback)" },
        @{File = "003_parameter_sensitivity_down.sql"; Desc = "003: Parameter Sensitivity (Rollback)" },
        @{File = "002_walk_forward_validation_down.sql"; Desc = "002: Walk-Forward Validation (Rollback)" }
    )
    
    foreach ($rollback in $rollbacks) {
        $filePath = [System.IO.Path]::Combine($MigrationsDir, $rollback.File)
        
        if (-not (Test-Path $filePath)) {
            Write-Host "  WARNING: Rollback file not found: $($rollback.File)" -ForegroundColor Yellow
            continue
        }
        
        $success = Invoke-SqlFile -FilePath $filePath -Description $rollback.Desc
        if (-not $success) {
            Write-Host "`nRollback failed. Stopping." -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "`nAll migrations rolled back successfully!" -ForegroundColor Green
}

# Function to verify schema
function Test-Schema {
    Write-Host "`nVerifying schema..." -ForegroundColor Cyan
    
    $expectedTables = @(
        "backtest_validations",
        "backtest_window_results",
        "parameter_sensitivity_results",
        "calibration_history",
        "drift_baselines",
        "drift_alerts",
        "performance_alerts",
        "retraining_jobs"
    )
    
    $allTablesExist = $true
    
    foreach ($table in $expectedTables) {
        $exists = Test-TableExists -TableName $table
        
        if ($exists) {
            Write-Host "  ✓ $table" -ForegroundColor Green
        }
        else {
            Write-Host "  ✗ $table (MISSING)" -ForegroundColor Red
            $allTablesExist = $false
        }
    }
    
    if ($allTablesExist) {
        Write-Host "`nSchema verification: PASSED!" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "`nSchema verification: FAILED!" -ForegroundColor Red
        return $false
    }
}

# Main execution
try {
    # Check psql is available
    $psqlCheck = Get-Command psql -ErrorAction SilentlyContinue
    if (-not $psqlCheck) {
        throw "psql command not found. Please ensure PostgreSQL client tools are installed."
    }
    
    # Test database connection
    Write-Host "Testing database connection..." -ForegroundColor Cyan
    $connTest = psql -h $Host -U $User -d $DatabaseName -c "SELECT 1;" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot connect to database. Please check credentials and ensure database exists."
    }
    Write-Host "  Connection successful!" -ForegroundColor Green
    Write-Host ""
    
    # Check PostgreSQL version and extensions (FIX ERR-001)
    Write-Host "Checking PostgreSQL version..." -ForegroundColor Cyan
    $versionQuery = "SHOW server_version_num;"
    $version = psql -h $Host -U $User -d $DatabaseName -t -c $versionQuery
    $version = $version.Trim()
    Write-Host "  PostgreSQL version: $version" -ForegroundColor Gray
    
    if ([int]$version -lt 130000) {
        Write-Warning "PostgreSQL 13+ recommended (detected: $version)"
        Write-Host "  Checking for pgcrypto extension..." -ForegroundColor Yellow
        $extQuery = "SELECT COUNT(*) FROM pg_extension WHERE extname='pgcrypto';"
        $hasPgcrypto = psql -h $Host -U $User -d $DatabaseName -t -c $extQuery
        $hasPgcrypto = $hasPgcrypto.Trim()
        
        if ($hasPgcrypto -eq "0") {
            Write-Host ""
            Write-Host "WARNING: pgcrypto extension not found!" -ForegroundColor Red
            Write-Host "Migrations require gen_random_uuid() function." -ForegroundColor Yellow
            Write-Host "Please run: CREATE EXTENSION pgcrypto;" -ForegroundColor Yellow
            Write-Host ""
            $createExt = Read-Host "Would you like to create the extension now? (y/N)"
            if ($createExt -eq 'y' -or $createExt -eq 'Y') {
                psql -h $Host -U $User -d $DatabaseName -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
                Write-Host "  pgcrypto extension created!" -ForegroundColor Green
            }
            else {
                throw "Cannot proceed without pgcrypto extension on PostgreSQL < 13"
            }
        }
        else {
            Write-Host "  pgcrypto extension found!" -ForegroundColor Green
        }
    }
    else {
        Write-Host "  PostgreSQL 13+ detected - gen_random_uuid() available!" -ForegroundColor Green
    }
    Write-Host ""
    
    # Execute requested action
    switch ($Action) {
        'Apply' {
            Apply-Migrations
        }
        'Rollback' {
            Rollback-Migrations
        }
        'Test' {
            Write-Host "Running FULL TEST (Apply + Verify + Rollback)..." -ForegroundColor Cyan
            Write-Host ""
            
            # Backup database
            Write-Host "Step 1: Creating backup..." -ForegroundColor Yellow
            $backupFile = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql"
            $backupPath = Join-Path $PSScriptRoot $backupFile
            pg_dump -h $Host -U $User $DatabaseName > $backupPath
            Write-Host "  Backup saved: $backupPath" -ForegroundColor Green
            Write-Host ""
            
            # Apply migrations
            Write-Host "Step 2: Applying migrations..." -ForegroundColor Yellow
            Apply-Migrations
            Write-Host ""
            
            # Verify schema
            Write-Host "Step 3: Verifying schema..." -ForegroundColor Yellow
            $verifySuccess = Test-Schema
            Write-Host ""
            
            if (-not $verifySuccess) {
                throw "Schema verification failed!"
            }
            
            # Rollback
            Write-Host "Step 4: Testing rollback..." -ForegroundColor Yellow
            Rollback-Migrations
            Write-Host ""
            
            # Verify rollback
            Write-Host "Step 5: Verifying rollback (tables should be gone)..." -ForegroundColor Yellow
            $tablesStillExist = $false
            foreach ($table in @("backtest_validations", "calibration_history")) {
                if (Test-TableExists -TableName $table) {
                    Write-Host "  ERROR: Table $table still exists after rollback!" -ForegroundColor Red
                    $tablesStillExist = $true
                }
            }
            
            if ($tablesStillExist) {
                throw "Rollback verification failed!"
            }
            
            Write-Host "  Rollback successful - tables removed!" -ForegroundColor Green
            Write-Host ""
            
            # Restore database
            Write-Host "Step 6: Restoring from backup..." -ForegroundColor Yellow
            psql -h $Host -U $User $DatabaseName < $backupPath
            Write-Host "  Database restored!" -ForegroundColor Green
            Write-Host ""
            
            Write-Host "===========================================" -ForegroundColor Cyan
            Write-Host "FULL TEST COMPLETED SUCCESSFULLY!" -ForegroundColor Green
            Write-Host "===========================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Migrations are verified and ready for production!" -ForegroundColor Green
        }
    }
}
catch {
    Write-Host ""
    Write-Host "===========================================" -ForegroundColor Red
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host "===========================================" -ForegroundColor Red
    exit 1
}
