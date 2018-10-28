import numpy as np

def activation(v):
    if (v>0 or v==0):
        result = 1
    else:
        result = -1
    return result

def predict(x,w):
    v = np.dot(x,w)
    output = activation(v)
    return output

def main():
    # the first column is for bias = 1
    pointA = np.array([1,0,1])
    pointB = np.array([1,1,0])
    labelA = 1
    labelB = -1

    w = np.array([0,0,0])
    eta = 0.5
    for i in range(0,3):
        errorA = labelA - predict(pointA, w)
        errorB = labelB - predict(pointB, w)
        dw = eta*errorA*pointA + eta*errorB*pointB
        w = w + dw

    print(labelA == predict(pointA, w))
    print(labelB == predict(pointB, w))

if __name__ == '__main__':
    main()
