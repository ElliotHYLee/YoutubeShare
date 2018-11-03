import numpy as np

def activation(v):
    # output 1 if v >=0, 0 otherwise
    output = None
    return output

def predict(x,w):
    # calculate logit, v
    v = None
    # activate v
    output = None
    return output

def main():
    #define input poitns
    # the first column is for bias = 1
    pointA = np.array([1,0.5,1])
    pointB = np.array([1,1,0.5])
    pointC = np.array([1,1,0.5])

    # define labels
    labelA = 1
    labelB = -1
    labelC = -1

    # define initial weights
    w = np.array([0,0,0])

    # define learning rate
    eta = 0.5

    for i in range(0,3):
        # this should be almost identiccal to example "LearningRule.py"
        print(w)

    print(labelA == predict(pointA, w))
    print(labelB == predict(pointB, w))
    print(labelC == predict(pointC, w))

if __name__ == '__main__':
    main()
