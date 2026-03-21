---
name: test-pipeline
description: Run the SimReady pipeline tests and verify USD output. Use when testing the CAD to USD conversion pipeline.
---

# Test Pipeline Skill
1. Run `python3 -m pytest tests/ -v`
2. If tests pass, verify the `run` subcommand is available: `python3 -m simready.cli run --help`
3. Report the test summary (pass/fail counts) and confirm the full pipeline command works:
   ```
   simready run "<query>" --output /tmp/test_output --sources github --max-results 3
   ```
   (Do not actually execute this — it makes live network calls. Just confirm the subcommand is registered and prints correct help.)
4. If any step fails, diagnose and suggest a fix
