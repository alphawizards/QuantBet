# QuantBet PostgreSQL Backup Script (PowerShell)
# Automated database backup with rotation policy

param(
    [int]$RetentionDays = 7
)

# Configuration
$BACKUP_DIR = ".\data\backups"
$DB_CONTAINER = "quantbet_db"
$DB_USER = "quantbet"
$DB_NAME = "quantbet"

# Create backup directory if it doesn't exist
if (!(Test-Path -Path $BACKUP_DIR)) {
    New-Item -ItemType Directory -Force -Path $BACKUP_DIR | Out-Null
}

# Generate timestamp
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$BACKUP_FILE = Join-Path $BACKUP_DIR "quantbet_$TIMESTAMP.sql"

# Perform backup
Write-Host "[$(Get-Date)] Starting database backup..."
try {
    docker exec $DB_CONTAINER pg_dump -U $DB_USER $DB_NAME | Out-File -FilePath $BACKUP_FILE -Encoding utf8
    
    # Compress backup
    Write-Host "[$(Get-Date)] Compressing backup..."
    Compress-Archive -Path $BACKUP_FILE -DestinationPath "$BACKUP_FILE.zip" -Force
    Remove-Item $BACKUP_FILE
    
    Write-Host "[$(Get-Date)] Backup completed: $BACKUP_FILE.zip"
    
    # Clean up old backups
    Write-Host "[$(Get-Date)] Cleaning up old backups..."
    $CutoffDate = (Get-Date).AddDays(-$RetentionDays)
    Get-ChildItem -Path $BACKUP_DIR -Filter "quantbet_*.zip" |
        Where-Object { $_.LastWriteTime -lt $CutoffDate } |
        Remove-Item -Force
    
    # Display backup info
    $BackupInfo = Get-Item "$BACKUP_FILE.zip"
    $BackupSize = "{0:N2} MB" -f ($BackupInfo.Length / 1MB)
    $BackupCount = (Get-ChildItem -Path $BACKUP_DIR -Filter "quantbet_*.zip").Count
    
    Write-Host "[$(Get-Date)] Backup successful!"
    Write-Host "  File: $($BackupInfo.Name)"
    Write-Host "  Size: $BackupSize"
    Write-Host "  Total backups retained: $BackupCount"
    
} catch {
    Write-Error "[$(Get-Date)] Backup failed: $_"
    exit 1
}
