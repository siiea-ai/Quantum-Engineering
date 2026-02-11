# Code Documentation Template

## Project: [Project Name]

---

## Overview

### Project Description

[2-3 paragraph description of what this code does, the problem it solves, and its main features]

### Quick Links

| Resource | Link |
|----------|------|
| Repository | |
| Documentation | |
| Issue Tracker | |
| DOI | |

---

## Installation

### Prerequisites

- Python >= 3.9
- [Other requirements]

### Quick Install

```bash
pip install [package-name]
```

### Development Install

```bash
git clone https://github.com/[username]/[repo].git
cd [repo]
pip install -e ".[dev]"
```

### Docker Install

```bash
docker pull [image-name]
docker run -it [image-name]
```

---

## Quick Start

### Basic Usage

```python
from package import MainClass

# Initialize
obj = MainClass(param1="value")

# Basic operation
result = obj.process(data)
print(result)
```

### Common Workflows

#### Workflow 1: [Description]

```python
# Step-by-step example
```

#### Workflow 2: [Description]

```python
# Step-by-step example
```

---

## API Reference

### Core Module: `package.core`

#### Class: `MainClass`

```python
class MainClass:
    """Main class for [functionality].

    This class provides [description of purpose and usage].

    Attributes:
        attr1 (type): Description.
        attr2 (type): Description.

    Example:
        >>> obj = MainClass(param1="value")
        >>> obj.method()
    """
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `__init__(param1, param2)` | Initialize instance | `None` |
| `method1(arg)` | Description | `type` |
| `method2(arg1, arg2)` | Description | `type` |

**Detailed Method Documentation:**

##### `method1(arg: type) -> return_type`

Description of what this method does.

**Parameters:**
- `arg` (type): Description of parameter

**Returns:**
- `return_type`: Description of return value

**Raises:**
- `ErrorType`: When condition

**Example:**
```python
result = obj.method1(value)
```

---

### Utilities Module: `package.utils`

#### Function: `utility_function`

```python
def utility_function(param1: type, param2: type = default) -> return_type:
    """One-line description.

    Longer description if needed.

    Args:
        param1: Description
        param2: Description (default: default_value)

    Returns:
        Description of return value

    Example:
        >>> utility_function(value1, value2)
    """
```

---

## Configuration

### Configuration File Format

```yaml
# config.yml
parameter1: value
parameter2:
  nested1: value
  nested2: value
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PACKAGE_CONFIG` | Path to config file | `./config.yml` |
| `PACKAGE_DEBUG` | Enable debug mode | `false` |

---

## Data Formats

### Input Data

#### Format 1: [Name]

```
Description of format
- Field 1: type, description
- Field 2: type, description
```

**Example:**
```json
{
  "field1": "value",
  "field2": 123
}
```

### Output Data

#### Format 1: [Name]

Description and structure of output.

---

## Examples

### Example 1: [Title]

**Goal:** What this example demonstrates

**Code:**
```python
# Complete working example
from package import MainClass

# Setup
obj = MainClass()

# Operation
result = obj.process(input_data)

# Output
print(result)
```

**Expected Output:**
```
[expected output here]
```

### Example 2: [Title]

[Another complete example]

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=package --cov-report=html

# Specific test file
pytest tests/test_module.py

# Specific test
pytest tests/test_module.py::test_function
```

### Test Categories

| Category | Command | Description |
|----------|---------|-------------|
| Unit | `pytest tests/unit/` | Fast, isolated tests |
| Integration | `pytest tests/integration/` | Component interaction |
| Slow | `pytest -m slow` | Long-running tests |

---

## Troubleshooting

### Common Issues

#### Issue: [Error message or symptom]

**Cause:** Why this happens

**Solution:**
```bash
# How to fix
```

#### Issue: [Another common issue]

**Cause:** [explanation]

**Solution:** [fix]

---

## Contributing

### Development Setup

```bash
git clone [repo]
cd [repo]
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Maximum line length: 88 characters

### Pull Request Process

1. Fork repository
2. Create feature branch
3. Write tests
4. Update documentation
5. Submit PR

---

## Changelog

### Version 1.0.0 (YYYY-MM-DD)

**Added:**
- Initial release
- Feature 1
- Feature 2

**Fixed:**
- Issue description

---

## License

[License type] - see [LICENSE](LICENSE) file.

---

## Citation

If you use this software, please cite:

```bibtex
@software{author2026,
  author = {LastName, FirstName},
  title = {Package Name},
  year = {2026},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/username/repo}
}
```

---

## Contact

- **Author:** [Name]
- **Email:** [email]
- **ORCID:** [0000-0000-0000-0000]

---

*Documentation last updated: [Date]*
