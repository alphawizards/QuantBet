# Database Migrations

This directory contains SQL migration files for the QuantBet database schema.

## Migration Naming Convention

- **Up migrations**: `NNN_description.sql` - Apply changes
- **Down migrations**: `NNN_description_down.sql` - Rollback changes
- Numbers are padded to 3 digits (001, 002, etc.)

## Migration Order

Migrations must be applied in sequential order:

1. `001_initial_schema.sql` - Base schema (already deployed)
2. `002_walk_forward_validation.sql` - Adds backtest validation tables
3. `003_parameter_sensitivity.sql` - Adds parameter sensitivity tracking
4. `004_transaction_costs.sql` - Adds transaction cost columns to existing tables
5. `005_monitoring_tables.sql` - Adds production monitoring tables

## Dependencies

- **002** has no dependencies (references initial schema only)
- **003** has no dependencies
- **004** DEPENDS on **002** (modifies backtest_validations table)
- **005** has no dependencies

## Running Migrations

### Apply All Migrations
```powershell
# From project root
.\scripts\test_migrations.ps1 -Action Apply
```

### Rollback All Migrations
```powershell
# From project root
.\scripts\test_migrations.ps1 -Action Rollback
```

### Apply Single Migration
```powershell
# Connect to database
psql -h localhost -U postgres -d quantbet

# Apply migration
\i src/database/schema.sql/migrations/002_walk_forward_validation.sql
```

### Rollback Single Migration
```powershell
psql -h localhost -U postgres -d quantbet
\i src/database/schema.sql/migrations/002_walk_forward_validation_down.sql
```

## Testing Migrations

Before deploying to production:

1. Backup database: `pg_dump quantbet > backup.sql`
2. Run test script: `.\scripts\test_migrations.ps1 -Action Test`
3. Verify schema: `psql -d quantbet -c "\dt"`
4. If successful, deploy to production
5. If failed, restore: `psql quantbet < backup.sql`

## Migration Best Practices

1. **Always create paired up/down scripts**
2. **Test rollback before deployment**
3. **Document breaking changes in comments**
4. **Use transactions for atomic operations**
5. **Add indexes after bulk data loads**

## Schema Validation

After applying migrations, verify:
- All tables exist
- Foreign keys are valid
- Indexes are created
- Constraints are enforced

```sql
-- Check all new tables
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check foreign key constraints
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint WHERE contype = 'f';
```

## Troubleshooting

### Migration fails with "table already exists"
- Check if migration already applied
- Run rollback script first
- Verify migration version number

### Foreign key constraint error
- Ensure parent table exists
- Check migration order
- Verify data integrity

### Permission denied
- Ensure database user has CREATE/ALTER privileges
- Run as postgres superuser if needed

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 002 | 2026-01-01 | Walk-forward validation tables |
| 003 | 2026-01-01 | Parameter sensitivity tracking |
| 004 | 2026-01-01 | Transaction cost columns |
| 005 | 2026-01-01 | Production monitoring tables |
