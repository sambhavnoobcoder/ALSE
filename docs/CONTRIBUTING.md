# Contributing to ALSE

We welcome contributions to the Adaptive Learned Segmentation Encoder project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alse.git
cd alse
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run before committing:
```bash
black .
flake8 .
mypy alse/
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Format code with Black
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Reporting Issues

Please use GitHub Issues to report bugs or request features.

Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## Code of Conduct

Be respectful and constructive. We aim to foster an inclusive community.

Thank you for contributing to ALSE!
