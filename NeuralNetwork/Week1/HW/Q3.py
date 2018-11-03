import numpy as np

# implement below functions
#======================================================================
def activation(v):
    return None

def predict(x,w):
    return None

def train(x,y, epochs=5):
    # define initial weight and eta
    w = None
    eta = None

    for epoch in range(0,epochs):
        dw_sum = None
        for i in range(0,x.shape[0]):
            input = None
            label = None
            error = None
            dw_sum += None
        w += None
    return w
#=======================================================================

def checkResults(x,y,w):
    for i in range(0,4):
        input = x[i,:]
        label = y[i]
        print(label == predict(input, w))

def AND_train(x):
    # label for AND
    y = np.array([0,0,0,1])
    # run 5 epochs for each of the 4 data points
    w = train(x,y,epochs=5)
    # check the result
    checkResults(x,y,w)

def OR_train(x):
    # label for OR
    y = np.array([0,1,1,1])
    # run 5 epochs for each of the 4 data points
    w = train(x,y,epochs=5)
    # check the result
    checkResults(x,y,w)

if __name__ == '__main__':
    # the first column is for bias = 1
    x = np.array([[1, 0,0],
                  [1, 0,1],
                  [1, 1,0],
                  [1, 1,1]])
    AND_train(x)
    OR_train(x)
