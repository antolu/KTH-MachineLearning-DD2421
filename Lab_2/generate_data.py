import numpy as np
import random

# Uncomment to generate the same random set each time
np.random.seed(64)

def getDefaultData() :
    classA = np.concatenate(
        (
            np.random.randn(20, 2) * 0.4 + [1.5, 0.5], 
            np.random.randn(20, 2) * 0.4 + [-1.5, 0.5]
        )
    )

    classB = np.random.randn(40, 2) * 0.4 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]), 
        -np.ones(classB.shape[0]))
    )

    N = inputs.shape[0] # Number of rows (samples)

    # Re-order samples
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return classA, classB, inputs, targets

def getNonLinearData() :
    classA = np.concatenate(
        (
            np.random.randn(10, 2) * 0.2 + [0.0, -1.5], 
            np.random.randn(10, 2) * 0.2 + [0.0, 1.5]
        )
    )

    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]), 
        -np.ones(classB.shape[0]))
    )

    N = inputs.shape[0] # Number of rows (samples)

    # Re-order samples
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return classA, classB, inputs, targets

def getSlackData() :
    classA = np.concatenate(
        (
            np.random.randn(20, 2) * 0.8 + [-1.0, 2.0], 
            np.random.randn(20, 2) * 0.8 + [1.0, 2.0]
        )
    )

    classB = np.random.randn(40, 2) * 0.8 + [0.0, -0.9]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate(
        (np.ones(classA.shape[0]), 
        -np.ones(classB.shape[0]))
    )

    N = inputs.shape[0] # Number of rows (samples)

    # Re-order samples
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return classA, classB, inputs, targets
