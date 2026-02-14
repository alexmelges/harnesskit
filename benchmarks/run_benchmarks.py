#!/usr/bin/env python3
"""
HarnessKit Benchmark Runner

Tests HarnessKit against a curated corpus of real-world LLM edit failures.
Reports success rates by category and overall metrics.
"""

import json
import os
import sys
import tempfile
import subprocess
import glob
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class BenchmarkResult:
    name: str
    category: str
    difficulty: int
    success: bool
    confidence: float
    match_type: str
    error: str
    execution_time_ms: float

@dataclass 
class CategoryStats:
    total: int
    passed: int
    failed: int
    avg_confidence: float
    avg_difficulty: float
    success_rate: float

def load_test_cases(benchmark_dir: str) -> List[Dict[str, Any]]:
    """Load all benchmark test cases from JSON files."""
    test_cases = []
    json_files = glob.glob(os.path.join(benchmark_dir, "*.json"))
    
    for file_path in sorted(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
                test_case['_file'] = os.path.basename(file_path)
                test_cases.append(test_case)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(test_cases)} test cases from {len(json_files)} files")
    return test_cases

def run_harnesskit(original: str, old_text: str, new_text: str, hk_path: str) -> Dict[str, Any]:
    """Run HarnessKit on a test case and return the result."""
    import time
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tmp') as tmp_file:
        tmp_file.write(original)
        tmp_file_path = tmp_file.name
    
    try:
        # Prepare edit instruction
        edit_data = {
            "file": tmp_file_path,
            "old_text": old_text,
            "new_text": new_text
        }
        
        # Run HarnessKit
        result = subprocess.run([
            sys.executable, hk_path, "apply", "--stdin"
        ], input=json.dumps(edit_data), capture_output=True, text=True)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            
            # Read the modified file to compare with expected
            with open(tmp_file_path, 'r') as f:
                actual_content = f.read()
            
            return {
                "success": output.get("status") == "applied",
                "confidence": output.get("confidence", 0.0),
                "match_type": output.get("match_type", "unknown"),
                "error": output.get("error", ""),
                "actual_content": actual_content,
                "execution_time_ms": execution_time
            }
        else:
            return {
                "success": False,
                "confidence": 0.0,
                "match_type": "none",
                "error": f"Exit code {result.returncode}: {result.stderr}",
                "actual_content": original,
                "execution_time_ms": execution_time
            }
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def run_single_test(test_case: Dict[str, Any], hk_path: str) -> BenchmarkResult:
    """Run a single benchmark test case."""
    original = test_case["original"]
    llm_edit = test_case["llm_edit"]
    expected = test_case["expected"]
    
    result = run_harnesskit(
        original=original,
        old_text=llm_edit["old_text"], 
        new_text=llm_edit["new_text"],
        hk_path=hk_path
    )
    
    # Check if result matches expected output
    success = (result["success"] and 
               result["actual_content"].strip() == expected.strip())
    
    return BenchmarkResult(
        name=test_case["name"],
        category=test_case["category"],
        difficulty=test_case["difficulty"],
        success=success,
        confidence=result["confidence"],
        match_type=result["match_type"],
        error=result["error"] if not success else "",
        execution_time_ms=result["execution_time_ms"]
    )

def calculate_category_stats(results: List[BenchmarkResult]) -> Dict[str, CategoryStats]:
    """Calculate statistics by failure category."""
    categories = {}
    
    for result in results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)
    
    stats = {}
    for category, cat_results in categories.items():
        total = len(cat_results)
        passed = sum(1 for r in cat_results if r.success)
        failed = total - passed
        avg_confidence = sum(r.confidence for r in cat_results) / total if total > 0 else 0
        avg_difficulty = sum(r.difficulty for r in cat_results) / total if total > 0 else 0
        success_rate = (passed / total * 100) if total > 0 else 0
        
        stats[category] = CategoryStats(
            total=total,
            passed=passed,
            failed=failed,
            avg_confidence=avg_confidence,
            avg_difficulty=avg_difficulty,
            success_rate=success_rate
        )
    
    return stats

def print_results(results: List[BenchmarkResult], category_stats: Dict[str, CategoryStats]):
    """Print detailed benchmark results."""
    print("\n" + "="*80)
    print("HARNESSKIT BENCHMARK RESULTS")
    print("="*80)
    
    # Overall stats
    total_tests = len(results)
    total_passed = sum(1 for r in results if r.success)
    total_failed = total_tests - total_passed
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    avg_execution_time = sum(r.execution_time_ms for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_passed} ({overall_success_rate:.1f}%)")
    print(f"  Failed: {total_failed}")
    print(f"  Avg Execution Time: {avg_execution_time:.1f}ms")
    
    # Category breakdown
    print(f"\nRESULTS BY CATEGORY:")
    print(f"{'Category':<25} {'Total':<6} {'Pass':<6} {'Fail':<6} {'Rate':<8} {'Confidence':<11} {'Difficulty':<10}")
    print("-" * 80)
    
    for category, stats in sorted(category_stats.items()):
        print(f"{category:<25} {stats.total:<6} {stats.passed:<6} {stats.failed:<6} "
              f"{stats.success_rate:>6.1f}% {stats.avg_confidence:>9.3f} "
              f"{stats.avg_difficulty:>8.1f}")
    
    # Failed tests details
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        print(f"\nFAILED TESTS DETAILS:")
        print("-" * 80)
        for result in failed_tests:
            print(f"❌ {result.name}")
            print(f"   Category: {result.category} | Difficulty: {result.difficulty}")
            print(f"   Error: {result.error}")
            print()
    
    # Match type distribution 
    match_types = {}
    for result in results:
        if result.success:
            match_types[result.match_type] = match_types.get(result.match_type, 0) + 1
    
    if match_types:
        print(f"\nMATCH TYPE DISTRIBUTION:")
        for match_type, count in sorted(match_types.items()):
            percentage = (count / total_passed * 100) if total_passed > 0 else 0
            print(f"  {match_type}: {count} ({percentage:.1f}%)")

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python run_benchmarks.py [--file <single_test.json>]")
        print("Run HarnessKit benchmarks against curated failure corpus")
        return
    
    # Find HarnessKit
    hk_path = os.path.join(os.path.dirname(__file__), "..", "hk.py")
    if not os.path.exists(hk_path):
        print("Error: Cannot find hk.py. Make sure you're running from benchmarks directory.")
        return 1
    
    benchmark_dir = os.path.dirname(__file__)
    
    # Load test cases
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        test_file = sys.argv[2]
        with open(test_file, 'r') as f:
            test_cases = [json.load(f)]
    else:
        test_cases = load_test_cases(benchmark_dir)
    
    if not test_cases:
        print("No test cases found!")
        return 1
    
    print(f"Running {len(test_cases)} benchmark tests...")
    
    # Run all tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i:2d}/{len(test_cases)}] {test_case['name'][:50]}...", end=" ")
        
        try:
            result = run_single_test(test_case, hk_path)
            results.append(result)
            status = "✅" if result.success else "❌"
            print(f"{status} ({result.confidence:.3f})")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(BenchmarkResult(
                name=test_case["name"],
                category=test_case["category"],
                difficulty=test_case["difficulty"],
                success=False,
                confidence=0.0,
                match_type="error",
                error=str(e),
                execution_time_ms=0.0
            ))
    
    # Calculate and display results
    category_stats = calculate_category_stats(results)
    print_results(results, category_stats)
    
    # Export results to JSON
    results_file = os.path.join(benchmark_dir, "benchmark_results.json")
    export_data = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "total_tests": len(results),
        "overall_success_rate": sum(1 for r in results if r.success) / len(results) * 100,
        "category_stats": {k: asdict(v) for k, v in category_stats.items()},
        "detailed_results": [asdict(r) for r in results]
    }
    
    with open(results_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nDetailed results exported to: {results_file}")
    
    # Return exit code based on success rate
    success_rate = sum(1 for r in results if r.success) / len(results)
    return 0 if success_rate >= 0.8 else 1  # 80% threshold for success

if __name__ == "__main__":
    sys.exit(main())