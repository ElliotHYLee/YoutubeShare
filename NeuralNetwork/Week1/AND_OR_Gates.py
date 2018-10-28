import numpy as np

def AND(x):
    return (x[1] and x[2])

def OR(x):
    return (x[1] or x[2])

def activation(v):
    if (v>0 or v==0):
        result = 1
    else:
        result = 0
    return result

def AND_Perceptron(x):
    w = np.array([-2, 1, 1])
    v = np.dot(x,w)
    return activation(v)

def OR_Perceptron(x):
    ## test your OR Gate
    return None

def main():
    # the first column is for bias = 1
    x = np.array([[1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])

    # Simple AND gate
    for i in range(0,4):
        a = AND(x[i,:])
        print(a)

    # Perceptron AND gate
    for i in range(0,4):
        a = AND_Perceptron(x[i,:])
        print(a)

    # Simple OR gate
    for i in range(0,4):
        a = OR(x[i,:])
        print(a)

    # Perceptron OR gate
    for i in range(0,4):
        a = OR_Perceptron(x[i,:])
        print(a)

if __name__ == '__main__':
    main()
