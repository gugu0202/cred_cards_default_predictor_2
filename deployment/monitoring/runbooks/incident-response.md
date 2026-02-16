# Incident Response Runbook for Credit Scoring System

## Overview
This runbook provides step-by-step procedures for responding to incidents in the Credit Scoring MLOps system.

## Incident Severity Levels

### SEV-1 (Critical)
- **Impact**: System completely down, no predictions possible
- **Response Time**: Immediate (within 15 minutes)
- **Resolution Time**: 2 hours
- **Examples**: 
  - All API instances down
  - Database completely unavailable
  - Security breach detected

### SEV-2 (High)
- **Impact**: Severe degradation, high error rates (>10%)
- **Response Time**: 30 minutes
- **Resolution Time**: 4 hours
- **Examples**:
  - 50% of pods failing
  - Response latency > 5 seconds
  - Data drift > 50%

### SEV-3 (Medium)
- **Impact**: Moderate degradation, some users affected
- **Response Time**: 2 hours
- **Resolution Time**: 24 hours
- **Examples**:
  - Single pod failures
  - Minor performance degradation
  - Warning alerts for resources

### SEV-4 (Low)
- **Impact**: Minimal impact, mostly internal
- **Response Time**: 1 business day
- **Resolution Time**: 1 week
- **Examples**:
  - Minor logging issues
  - Non-critical metric alerts
  - Documentation updates

## Incident Response Process

### Phase 1: Detection and Triage
1. **Alert Receipt**
   - Monitor alerts in #alerts-production Slack channel
   - Check PagerDuty for critical alerts
   - Verify alert in Grafana/Prometheus

2. **Initial Assessment**
   ```bash
   # Check system status
   kubectl get pods -n credit-scoring-production
   kubectl get svc -n credit-scoring-production
   kubectl get hpa -n credit-scoring-production
   
   # Check logs
   kubectl logs -f deployment/credit-scoring-api -n credit-scoring-production
   kubectl logs -f deployment/credit-scoring-api --previous -n credit-scoring-production