# Convolutional Neural Network Project
The goal of the project is to create a tool to recognize and classify hand-written digits from an image using a convolutional neural network.

I am continuing this project from period 1. 

## Documentation
- [Weekly report](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/Weekly_reports/week_4.md)
- [Requirement Specification](https://https://github.com/jooniku/digit_recognition_project/)
- [Testing document](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/testing_document.md)
- [Implementation document](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/implementation_document.md)
- [Changelog](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/changelog.md)

## Installation
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
