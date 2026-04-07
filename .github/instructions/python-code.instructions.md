---
applyTo: "**/*.py"
---

# Python Coding Standards

## Style

- Format all code with **Black** (default line length: 88).
- Sort imports with **isort** (profile: black).
- Use `snake_case` for functions, methods, variables, and modules. Never use camelCase.
- Use `UPPER_SNAKE_CASE` for constants.
- Use `PascalCase` only for class names.

## Conciseness & Explicitness

- Write concise code. Avoid unnecessary abstraction, wrapper functions, or indirection.
- Be explicit over implicit: no magic, no hidden state, no surprising defaults.
- Prefer flat over nested. If deeply indented, consider extracting helper functions or using early returns.
- Avoid comments that restate the code. Comment only to explain *why*, never *what*.
- Keep functions short — one clear purpose each.
- In PyTorch code, always annotate tensor dimensions as inline comments (e.g. `# (batch, channels, height, width)`).

## Type Hints

- Use type hints on all function signatures (parameters and return types).
- Use `from __future__ import annotations` for modern annotation syntax.
- Prefer built-in generics (`list[str]`, `dict[str, int]`) over `typing` equivalents.

## Naming

- Prefer short, clear names. Brevity wins when meaning is obvious from context.
- Single-letter names are fine in tight loops, lambdas, and well-known conventions (e.g. `x` for tensors, `i` for indices, `n` for counts).
- Boolean variables and functions should read as assertions: `is_valid`, `has_access`, `can_retry`.
- Avoid abbreviations unless they are universally understood (`url`, `id`, `config`).

## Error Handling

- Catch specific exceptions. Never use bare `except:` or `except Exception:` without good reason.
- Fail fast: validate inputs early and raise clear errors.

## Imports

- Relative imports within the same package are fine. Use absolute imports for third-party and stdlib.
- Group imports: stdlib → third-party → local, separated by blank lines (isort handles this).

## Testing

- Use **pytest** for all tests. Do not use unittest.
- Name test files `test_<module>.py` and test functions `test_<behavior>`.
- Keep tests focused: one assertion per test when practical.
- Use fixtures for shared setup. Prefer factory fixtures over complex shared state.
- Use `pytest.raises` for expected exceptions, `pytest.mark.parametrize` for variant inputs.
- No test should depend on execution order or external state (database, network) unless explicitly marked with an appropriate marker.

## General

- Prefer f-strings for string formatting.
- Use pathlib over os.path for file system operations.
- Use dataclasses or attrs for plain data containers; avoid raw dicts for structured data.
- Write docstrings for public functions and classes (Google style).
- No dead code. No commented-out code in commits.
