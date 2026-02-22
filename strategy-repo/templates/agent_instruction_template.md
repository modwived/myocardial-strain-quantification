# Agent: Phase N — [Module Name]

## Mission
[One paragraph: what this agent builds, why it matters, and how it fits into the larger system.]

## Status: IN PROGRESS

## Files You Own
List every file this agent is responsible for. Mark whether it already exists or needs creating.

- `path/to/file_a.py` — [description] **(EXISTS — enhance)**
- `path/to/file_b.py` — [description] **(CREATE)**
- `path/to/file_c.py` — [description] **(CREATE)**
- `tests/test_module.py` — unit tests **(CREATE)**
- `configs/module.yaml` — configuration **(CREATE)**

## Detailed Requirements

### 1. file_a.py — Enhance existing skeleton
What already exists (checked) and what to add (unchecked):
- [x] `existing_function(arg)` — already implemented
- [ ] `new_function(arg: Type) -> ReturnType` — [describe behavior, edge cases]
- [ ] `another_function(arg: Type) -> ReturnType` — [describe behavior]
- [ ] Add error handling for [specific scenarios]
- [ ] Handle edge case: [describe]

### 2. file_b.py — Create new
```python
def function_one(data: InputType) -> OutputType:
    """[What it does].

    Args:
        data: [Description].

    Returns:
        [Description].
    """

def function_two(config: dict) -> SomeObject:
    """[What it does]."""
```
- [ ] Implement function_one with [specific algorithm/approach]
- [ ] Implement function_two with [specific logic]
- [ ] Handle edge case: [describe]

### 3. configs/module.yaml
```yaml
parameter_a: default_value
parameter_b: default_value
nested:
  setting_one: value
  setting_two: value
```

### 4. tests/test_module.py
- Test `function_one` with valid input → expected output
- Test `function_one` with edge case → expected behavior
- Test `function_two` returns correct type
- Test error handling for invalid inputs
- Use synthetic/mock data only — no external dependencies

## Interface Contract

### What this module receives (inputs):
```python
# From Phase X:
input_data: dict  # {"key": value_type, ...}

# From Phase Y (optional):
optional_config: dict | None
```

### What this module produces (outputs):
```python
# To Phase Z:
output_data: dict  # {"key": value_type, ...}

# Example:
{
    "result": float,
    "metadata": {"timestamp": str, "version": str},
    "details": [{"item": str, "score": float}, ...],
}
```

## Dependencies
List any packages this module needs beyond what's in the project's requirements:
```
package-name>=1.0.0    # What it's used for
another-package>=2.0.0 # What it's used for
```

## If You Get Stuck
- [Link to relevant documentation]
- [Common pitfall and how to avoid it]
- [Debugging hint specific to this module]
- If [specific error], check [specific cause]
- If [metric] is wrong, verify [specific assumption]
