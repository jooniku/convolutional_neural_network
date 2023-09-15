# Digit Recognition Project
The goal of the project is to create a tool to recognize and classify hand-written digits from an image using a convolutional neural network.

## Documentation
- [Weekly report 1](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/Weekly_reports/week_1.md)
- [Requirement Specification](https://https://github.com/jooniku/digit_recognition_project/)
- [Changelog](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/changelog.md)

## Installation (currently does not work)
Poetry must be installed on the system.

1. Install dependencies:
```bash
poetry install
```
2. Run the application:
```bash
poetry run invoke start
```

## Testing

Run tests:
```bash
poetry run invoke test
```
Generate the coverage report:
```bash
poetry run invoke coverage-report
```
_The report will be in the htmlcov directory_

### Test code quality using pylint:
```bash
poetry run invoke lint
```
