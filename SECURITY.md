# Security Policy

## Scope

This is an educational curriculum repository containing markdown lesson files and Python code examples. It does not run any services or handle user data directly.

## Reporting a Vulnerability

If you discover a security concern — such as embedded credentials, malicious code in examples, or links to compromised resources — please report it responsibly:

**Email:** security@siiea.ai

### What to Include

- Description of the concern
- File path and line number(s)
- Steps to reproduce (if applicable)
- Suggested fix (if you have one)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Assessment:** Within 1 week
- **Resolution:** Depends on severity, typically within 2 weeks

## What We Consider Security Issues

- Embedded API keys, tokens, or credentials
- Malicious or obfuscated code in computational examples
- Links to compromised or malicious external resources
- Personal information that should not be public

## What Is NOT a Security Issue

- Bugs in example Python code (use regular issues)
- Mathematical errors (use regular issues)
- Broken links to legitimate resources (use regular issues)

## Python Code Safety

All Python examples in this curriculum use standard scientific libraries (`numpy`, `scipy`, `matplotlib`, `sympy`). Before running any code:

1. Review the code to understand what it does
2. Run in an isolated environment (virtualenv, conda, or container)
3. Code examples are educational — they are not production software

## Supported Versions

| Version | Supported |
|---------|-----------|
| Current (main branch) | Yes |
| Older commits | No |
