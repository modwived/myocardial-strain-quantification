# Post-Agent Integration Checklist

Run through this checklist after all parallel agents have completed their work.

---

## 1. Survey the Damage

```bash
# What changed?
git status --short

# How much changed?
git diff --stat

# Any file touched by multiple agents? (should be zero)
git diff --stat | sort | uniq -d
```

**If two agents modified the same file:** resolve manually before proceeding.

---

## 2. Fix Build Issues

### Check imports
```bash
# Python: verify all packages are importable
python -c "import module_a; import module_b; import module_c"

# Node: verify no missing deps
npm install && npm run build

# Rust: verify compilation
cargo build
```

### Check dependencies
- Did any agent add a new package? Update requirements.txt / package.json
- Are there version conflicts between agents' additions?

### Check gitignore
- Did a broad pattern (e.g., `data/`) accidentally match source files (e.g., `src/data/`)?
- Fix with anchored patterns: `/data/` instead of `data/`

---

## 3. Run Tests

```bash
# Run the full suite
python -m pytest tests/ -v --tb=short

# Or for other ecosystems:
npm test
cargo test
go test ./...
```

### Common failure patterns and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import X` | Missing `__init__.py` or circular import | Add init file or restructure imports |
| `TypeError: expected dict, got tuple` | Interface mismatch between agents | Align return types to the interface contract |
| `RuntimeError: size mismatch` | Tensor/array shape incompatibility | Check that producer and consumer agree on dimensions |
| `ModuleNotFoundError` | Agent used a package not in requirements | Add to requirements and install |
| `KeyError: 'expected_key'` | Agent A outputs different keys than Agent B expects | Align to the interface contract |

---

## 4. Test at the Seams

The code *within* each module is usually correct. Focus on **module boundaries**:

```python
# Test that Phase 1 output feeds correctly into Phase 2
phase1_output = phase1.process(test_input)
phase2_result = phase2.consume(phase1_output)  # Does this work?

# Test the full pipeline end-to-end
result = pipeline.run(synthetic_input)
assert result is not None
assert "expected_key" in result
```

---

## 5. Commit

### Option A: One commit per phase (cleaner history)
```bash
git add module_a/ tests/test_module_a.py
git commit -m "Implement Phase 1: Data Pipeline"

git add module_b/ tests/test_module_b.py
git commit -m "Implement Phase 2: Model Training"
```

### Option B: Single commit (simpler)
```bash
git add -A
git commit -m "Implement all N phases via parallel agents"
```

---

## 6. Validate End-to-End

```bash
# Generate synthetic test data if real data isn't available
python scripts/generate_test_data.py

# Run the full pipeline
python -m main --input test_data/ --output results/

# Check results are reasonable
python scripts/validate_results.py results/
```

---

## 7. Document What Happened

Update the agent instruction files with:
- [ ] What was actually implemented (vs what was planned)
- [ ] Any bugs found and fixed during integration
- [ ] Interface changes made during integration
- [ ] Remaining TODOs for the next iteration
