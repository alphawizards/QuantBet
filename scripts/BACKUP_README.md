# Database Backup Scripts

## Quick Start

### Windows (PowerShell)
```powershell
# Run manual backup
.\scripts\backup_db.ps1

# Run with custom retention (keep 14 days)
.\scripts\backup_db.ps1 -RetentionDays 14
```

### Linux/Mac (Bash)
```bash
# Make executable
chmod +x scripts/backup_db.sh

# Run manual backup
./scripts/backup_db.sh
```

## Scheduled Backups

### Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Name: "QuantBet DB Backup"
4. Trigger: Daily at 2:00 AM
5. Action: Start a program
   - Program: `powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\Users\ckr_4\01 Projects\QuantBet\QuantBet\scripts\backup_db.ps1"`
6. Finish

### Linux/Mac Crontab
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /path/to/QuantBet && ./scripts/backup_db.sh >> ./data/backups/backup.log 2>&1
```

## Backup Location
- **Directory**: `./data/backups/`
- **Format**: `quantbet_YYYYMMDD_HHMMSS.sql.gz` (Linux) or `.zip` (Windows)
- **Retention**: 7 days by default

## Restore Database

### From Backup
```bash
# Linux/Mac
gunzip -c data/backups/quantbet_20251231_020000.sql.gz | docker exec -i quantbet_db psql -U quantbet quantbet

# Windows
Expand-Archive data\backups\quantbet_20251231_020000.sql.zip -DestinationPath temp
Get-Content temp\quantbet_20251231_020000.sql | docker exec -i quantbet_db psql -U quantbet quantbet
```

## Best Practices
1. **Test Restores**: Periodically test backup restoration
2. **Off-site Backups**: Copy to cloud storage (AWS S3, Google Drive, etc.)
3. **Monitor Space**: Ensure backup directory has adequate space
4. **Verify Integrity**: Check backup file sizes are reasonable
