"""
Comprehensive Test Runner for QuantBet.

Runs all test suites in sequence and generates reports:
- Unit tests with coverage
- Integration tests
- Performance benchmarks
- Security tests
- Smoke tests

Outputs:
- HTML coverage report
- JSON results summary
- Performance metrics
- Test execution summary
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class TestRunner:
    """Orchestrates test execution and reporting."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'security_tests': {},
            'smoke_tests': {},
            'failed': False
        }
        self.project_root = Path(__file__).parent.parent
    
    def run_command(self, cmd: List[str], name: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            return {
                'name': name,
                'command': ' '.join(cmd),
                'exit_code': result.returncode,
                'duration_seconds': round(duration, 2),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'name': name,
                'command': ' '.join(cmd),
                'exit_code': -1,
                'duration_seconds': 300,
                'error': 'Timeout after 300 seconds',
                'passed': False
            }
        except Exception as e:
            return {
                'name': name,
                'command': ' '.join(cmd),
                'exit_code': -1,
                'error': str(e),
                'passed': False
            }
    
    def run_unit_tests(self):
        """Run unit tests with coverage."""
        print("\n🔬 RUNNING UNIT TESTS")
        
        result = self.run_command([
            sys.executable, '-m', 'pytest',
            'tests/unit/',
            '-v',
            '--tb=short',
            '--cov=src',
            '--cov-report=html:coverage_html',
            '--cov-report=json:coverage.json',
            '--cov-report=term',
            '-m', 'unit',
            '--no-cov-on-fail'
        ], 'Unit Tests')
        
        self.results['unit_tests'] = result
        
        # Parse coverage if available
        coverage_file = self.project_root / 'coverage.json'
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
                self.results['unit_tests']['coverage'] = {
                    'total_coverage': coverage_data['totals']['percent_covered'],
                    'lines_covered': coverage_data['totals']['covered_lines'],
                    'lines_total': coverage_data['totals']['num_statements']
                }
        
        if not result['passed']:
            self.results['failed'] = True
        
        return result['passed']
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n🔗 RUNNING INTEGRATION TESTS")
        
        result = self.run_command([
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v',
            '--tb=short',
            '-m', 'integration',
            '--no-cov'
        ], 'Integration Tests')
        
        self.results['integration_tests'] = result
        
        if not result['passed']:
            self.results['failed'] = True
        
        return result['passed']
    
    def run_performance_tests(self):
        """Run performance benchmarks."""
        print("\n🚀 RUNNING PERFORMANCE TESTS")
        
        result = self.run_command([
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v',
            '--tb=short',
            '-m', 'integration',
            '-k', 'performance',
            '--no-cov'
        ], 'Performance Tests')
        
        self.results['performance_tests'] = result
        
        # Don't fail overall if performance tests fail (warnings only)
        return True
    
    def run_security_tests(self):
        """Run security tests."""
        print("\n🔒 RUNNING SECURITY TESTS")
        
        result = self.run_command([
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v',
            '--tb=short',
            '-m', 'security',
            '--no-cov'
        ], 'Security Tests')
        
        self.results['security_tests'] = result
        
        if not result['passed']:
            self.results['failed'] = True
        
        return result['passed']
    
    def run_smoke_tests(self):
        """Run smoke tests."""
        print("\n💨 RUNNING SMOKE TESTS")
        
        result = self.run_command([
            sys.executable, '-m', 'pytest',
            'tests/smoke/',
            '-v',
            '--tb=short',
            '-m', 'smoke',
            '--no-cov'
        ], 'Smoke Tests')
        
        self.results['smoke_tests'] = result
        
        if not result['passed']:
            self.results['failed'] = True
        
        return result['passed']
    
    def generate_summary(self):
        """Generate test execution summary."""
        print("\n" + "="*80)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*80)
        
        total_duration = sum(
            r.get('duration_seconds', 0)
            for r in [
                self.results['unit_tests'],
                self.results['integration_tests'],
                self.results['performance_tests'],
                self.results['security_tests'],
                self.results['smoke_tests']
            ]
            if r
        )
        
        summary = {
            'timestamp': self.results['timestamp'],
            'total_duration_seconds': round(total_duration, 2),
            'overall_status': 'PASSED' if not self.results['failed'] else 'FAILED',
            'test_suites': {}
        }
        
        for suite_name in ['unit_tests', 'integration_tests', 'performance_tests', 'security_tests', 'smoke_tests']:
            suite = self.results[suite_name]
            if suite:
                summary['test_suites'][suite_name] = {
                    'status': 'PASSED' if suite.get('passed') else 'FAILED',
                    'duration_seconds': suite.get('duration_seconds', 0),
                    'exit_code': suite.get('exit_code', -1)
                }
        
        # Add coverage info
        if self.results['unit_tests'].get('coverage'):
            summary['coverage'] = self.results['unit_tests']['coverage']
        
        self.results['summary'] = summary
        
        # Print summary
        print(f"\nOverall Status: {summary['overall_status']}")
        print(f"Total Duration: {summary['total_duration_seconds']}s")
        print(f"\nTest Suite Results:")
        for suite_name, suite_data in summary['test_suites'].items():
            status = "✅" if suite_data['status'] == 'PASSED' else "❌"
            print(f"  {status} {suite_name}: {suite_data['status']} ({suite_data['duration_seconds']}s)")
        
        if summary.get('coverage'):
            cov = summary['coverage']
            print(f"\n📈 Coverage: {cov['total_coverage']:.1f}% ({cov['lines_covered']}/{cov['lines_total']} lines)")
        
        print("\n" + "="*80)
        
        return summary
    
    def save_results(self):
        """Save results to JSON file."""
        results_file = self.project_root / 'test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Also save just the summary
        summary_file = self.project_root / 'test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results['summary'], f, indent=2)
        
        print(f"💾 Summary saved to: {summary_file}")
    
    def run_all(self):
        """Run all test suites."""
        print("\n" + "="*80)
        print("🧪 QUANTBET COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        # Run tests in sequence
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_tests()
        self.run_security_tests()
        self.run_smoke_tests()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Exit with appropriate code
        if self.results['failed']:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
        else:
            print("\n✅ ALL TESTS PASSED")
            sys.exit(0)


def main():
    """Main entry point."""
    runner = TestRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
