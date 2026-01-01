# Deployment Gates Checklist

**Purpose**: Ensure quality and safety before deploying to production.

---

## Pre-Deployment Gates

### Testing ✅
- [ ] **Unit Tests**: 100% passing, coverage ≥ 80%
- [ ] **Integration Tests**: 100% passing
- [ ] **E2E Tests**: 100% passing  
- [ ] **Smoke Tests**: 100% passing on staging
- [ ] **Performance Tests**: P95 response time < 500ms
- [ ] **Load Tests**: System stable under expected load
- [ ] **Security Scan**: No critical or high vulnerabilities

### Code Quality ✅
- [ ] **Code Review**: Minimum 2 approvals required
- [ ] **Linting**: No errors, warnings addressed
- [ ] **Type Checking**: TypeScript/mypy passing (if applicable)
- [ ] **Documentation**: All public APIs documented

### Database ✅
- [ ] **Migrations Tested**: Run successfully on staging database
- [ ] **Migration Validation**: Data loss check passed
- [ ] **Rollback Script**: Tested and ready
- [ ] **Backup Created**: Pre-deployment backup taken

### Operations ✅
- [ ] **Rollback Plan**: Documented and rehearsed
- [ ] **Monitoring Dashboards**: Created and tested
- [ ] **Alert Thresholds**: Configured appropriately
- [ ] **On-Call Engineer**: Assigned and available
- [ ] **Deployment Window**: Scheduled (prefer low-traffic hours)

### Stakeholder Sign-Off ✅
- [ ] **Product Owner**: Feature demo approved
- [ ] **QA Engineer**: Testing sign-off provided
- [ ] **Tech Lead**: Architecture review passed
- [ ] **DevOps**: Infrastructure review passed

---

## During Deployment

### Deployment Steps
1. [ ] Announce deployment in #engineering channel
2. [ ] Put system in maintenance mode (if downtime required)
3. [ ] Take database backup
4. [ ] Run database migrations
5. [ ] Deploy backend application
6. [ ] Deploy frontend application
7. [ ] Run smoke tests
8. [ ] Remove maintenance mode
9. [ ] Monitor for 15 minutes

---

## Post-Deployment Gates

### Immediate Verification (0-15 min)
- [ ] **Smoke Tests**: All passing in production
- [ ] **Error Rate**: < 0.1% for first 15 minutes
- [ ] **API Response Time**: P95 < 500ms
- [ ] **Frontend Loads**: Homepage accessible
- [ ] **Critical Journey**: User can view today's picks

### Short-Term Monitoring (15-60 min)
- [ ] **Error Rate**: Remains < 0.1%
- [ ] **Response Time**: P95 remains < 500ms
- [ ] **No Error Spikes**: In error tracking (Sentry)
- [ ] **Database Performance**: No slow queries (> 1s)
- [ ] **Memory/CPU Usage**: Within normal ranges

### Extended Monitoring (1-24 hours)
- [ ] **Error Rate**: < 0.05% sustained
- [ ] **User Reports**: No critical bugs reported
- [ ] **Performance**: No degradation over time
- [ ] **Data Quality**: Predictions generating correctly

---

## Rollback Criteria

**IMMEDIATE ROLLBACK** if any of the following occur:

### Critical Issues (Rollback Immediately)
- [ ] **Error Rate > 1%** for any 5-minute period
- [ ] **API Response Time P95 > 2 seconds** sustained
- [ ] **Database Migration Failed** or data corruption detected
- [ ] **Critical Functionality Broken**: Users cannot view predictions
- [ ] **Security Vulnerability** introduced
- [ ] **Data Loss**: Any data deleted or corrupted

### Major Issues (Rollback if not fixed in 15 min)
- [ ] **Error Rate > 0.5%** sustained
- [ ] **API Response Time P95 > 1 second** sustained
- [ ] **Frontend Not Loading** for some users
- [ ] **Database Connection** issues

### Rollback Procedure
1. [ ] Announce rollback in #engineering
2. [ ] Revert backend deployment to previous version
3. [ ] Revert frontend deployment to previous version
4. [ ] Run rollback migration (if DB changes made)
5. [ ] Restore database from backup (if needed)
6. [ ] Run smoke tests on rolled-back version
7. [ ] Post-mortem: Document what went wrong

---

## Deployment Approval

**Pre-Deployment Sign-Off**:
- [ ] Deployer: ___________________ Date: ______
- [ ] QA Lead: ___________________ Date: ______
- [ ] Tech Lead: _________________ Date: ______

**Post-Deployment Sign-Off**:
- [ ] Deployment Successful: ☐ Yes ☐ No (rolled back)
- [ ] Smoke Tests Passed: ☐ Yes ☐ No
- [ ] Monitoring Stable (1hr): ☐ Yes ☐ No
- [ ] Verified By: ___________________ Date/Time: ______

---

## Notes

**Deployment Date**: ________________  
**Version Deployed**: ________________  
**Issues Encountered**: 

**Post-Deployment Actions Needed**:
