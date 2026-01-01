#!/bin/bash
# QuantBet PostgreSQL Backup Script
# Automated database backup with rotation policy

set -e  # Exit on error

# Configuration
BACKUP_DIR="./data/backups"
DB_CONTAINER="quantbet_db"
DB_USER="quantbet"
DB_NAME="quantbet"
RETENTION_DAYS=7

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/quantbet_${TIMESTAMP}.sql"

# Perform backup
echo "[$(date)] Starting database backup..."
docker exec ${DB_CONTAINER} pg_dump -U ${DB_USER} ${DB_NAME} > "${BACKUP_FILE}"

# Compress backup
echo "[$(date)] Compressing backup..."
gzip "${BACKUP_FILE}"

echo "[$(date)] Backup completed: ${BACKUP_FILE}.gz"

# Clean up old backups (keep only last RETENTION_DAYS days)
echo "[$(date)] Cleaning up old backups..."
find "${BACKUP_DIR}" -name "quantbet_*.sql.gz" -mtime +${RETENTION_DAYS} -delete

# Display backup info
BACKUP_SIZE=$(du -h "$ {BACKUP_FILE}.gz" | cut -f1)
BACKUP_COUNT=$(find "${BACKUP_DIR}" -name "quantbet_*.sql.gz" | wc -l)

echo "[$(date)] Backup successful!"
echo "  File: ${BACKUP_FILE}.gz"
echo "  Size: ${BACKUP_SIZE}"
echo "  Total backups retained: ${BACKUP_COUNT}"

# Optional: Upload to cloud storage (uncomment if needed)
# aws s3 cp "${BACKUP_FILE}.gz" s3://your-bucket/quantbet-backups/
# rclone copy "${BACKUP_FILE}.gz" remote:quantbet-backups/
