# Contributing to Model Monitoring Tool (MMT)

Thank you for your interest in contributing to the Model Monitoring Tool! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful, inclusive, and considerate in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one.

When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if applicable
- Include details about your configuration and environment

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- Use a clear and descriptive title
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful to most users
- List any relevant examples to demonstrate the enhancement
- Include any references or resources that support your suggestion

### Pull Requests

- Fill in the required template
- Follow the coding style and conventions used in the project
- Include appropriate test cases
- Update the documentation to reflect any changes
- Ensure all tests pass before submitting your pull request
- Link the pull request to any related issues

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- npm 8 or higher

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mmt.git
   cd mmt
   ```

2. Create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask development server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Coding Guidelines

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions, classes, and modules
- Keep functions focused on a single responsibility
- Use meaningful variable and function names

### JavaScript/TypeScript Code Style

- Follow the ESLint configuration provided in the project
- Use TypeScript types and interfaces
- Write JSDoc comments for functions and components
- Use functional components and hooks in React
- Follow the project's component structure and naming conventions

### Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage, especially for critical functionality
- Include both positive and negative test cases

## Git Workflow

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes with clear, descriptive commit messages
6. Push your branch to your fork
7. Submit a pull request to the `main` branch of the original repository

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Release Process

1. Version numbers follow [Semantic Versioning](https://semver.org/)
2. Changes for each release are documented in the CHANGELOG.md file
3. Releases are tagged in the repository using the version number

## Questions?

If you have any questions about contributing, please open an issue or contact the project maintainers.

Thank you for contributing to the Model Monitoring Tool!
