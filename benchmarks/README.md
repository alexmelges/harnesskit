# HarnessKit Benchmark Dataset

## Overview

This benchmark dataset contains real-world LLM code edit failures that HarnessKit is designed to solve. Each test case demonstrates a common failure mode where exact string matching fails, but fuzzy matching should succeed.

## Dataset Structure

Each benchmark file is a JSON document containing:

- `name`: Human-readable test case name
- `category`: Failure type category  
- `difficulty`: 1-5 rating (1=easy, 5=hard)
- `source`: Where this failure pattern was observed
- `description`: What the LLM was trying to do
- `original`: The original file content
- `llm_edit`: The LLM's attempted edit (contains the error)
- `expected`: The correct result after applying the edit
- `metadata`: Additional context about the failure

## Failure Categories

### 1. **whitespace_mismatch**
LLM produces slightly different whitespace (extra spaces, mixed spacing) but semantically identical code.

### 2. **stale_context** 
LLM references old content that has been modified since it last read the file.

### 3. **partial_match**
LLM only matches part of the intended code block, missing surrounding context.

### 4. **indentation_drift**
Inconsistent indentation, mixed tabs/spaces, or incorrect nesting levels.

### 5. **line_number_off**
When using line-number-based edits, LLM references wrong line numbers.

### 6. **encoding_issues**
Character encoding differences, invisible characters, or Unicode normalization problems.

## Running Benchmarks

Use the benchmark runner to test HarnessKit against all cases:

```bash
python run_benchmarks.py
```

This will:
1. Load all test cases from `*.json` files
2. Apply the `llm_edit` using HarnessKit
3. Compare result with `expected` output
4. Report success rate by category and overall

## Test Case Format

```json
{
  "name": "Aider Rust visibility modifier",
  "category": "whitespace_mismatch", 
  "difficulty": 2,
  "source": "GitHub issue Aider-AI/aider#827",
  "description": "LLM tries to add pub keyword but whitespace doesn't match exactly",
  "original": "struct Shards {\n    len: u32,\n    arr_offset: u64,\n}",
  "llm_edit": {
    "old_text": "struct Shards {\n len: u32,\n arr_offset: u64,\n}",
    "new_text": "pub struct Shards {\n len: u32,\n arr_offset: u64,\n}"
  },
  "expected": "pub struct Shards {\n    len: u32,\n    arr_offset: u64,\n}",
  "metadata": {
    "model": "claude-3.5-sonnet",
    "tool": "aider",
    "language": "rust",
    "common_pattern": true
  }
}
```

## Success Metrics

- **Pass Rate**: Percentage of test cases where HarnessKit successfully applies the edit
- **Exact Match**: Result exactly matches expected output
- **Semantic Match**: Result is functionally equivalent (for future enhancement)
- **Confidence Score**: Average confidence reported by HarnessKit
- **Category Breakdown**: Success rates by failure type

## Adding Test Cases

To add new test cases:

1. Find a real failure example from GitHub issues, forums, or personal experience
2. Create a JSON file in the appropriate category subdirectory
3. Include original failing edit attempt and correct expected result
4. Test locally with `python run_benchmarks.py --file your_test.json`
5. Verify HarnessKit handles it correctly

The goal is to build a comprehensive corpus that proves HarnessKit can solve real problems that plague LLM code editing tools.