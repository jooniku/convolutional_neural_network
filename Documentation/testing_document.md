# Testing Documentation

### How is the program tested
- Testing is done with Unittests
- When appropriate, varied inputs are used
- Computing functions are tested with correct, hand-computed results
- The dataset and data extraction from it is __not__ tested
- The app that the user interacts with is __not__ tested

### Other tests
- Network can achieve 0-loss with a single input. It is tested within the unittests.
- Gradient checks are performed (within unittests) to have a high probability of correct backpropagation

### Code quality
- Pylint is used to check code quality

### Test coverage
- Test coverage is 84 %.
- Test coverage doesn't seem to be the best indicator of a working project with neural networks.
- Therefore the other tests have been created.
- Plotting has not been tested.
- Or data loading.

### Test report
![coverage report](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/images/cvrg_report_cnn.png)
