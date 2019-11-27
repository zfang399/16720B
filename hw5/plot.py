import numpy as np
import matplotlib.pyplot as plt

def sigmoid_d(x):
    return np.exp(-x)/np.square(1 + np.exp(-x))

def tanh_d(x):
    return 1 - np.square(np.tanh(x))
def main():
    """Main function
    :returns: TODO

    """
    x = np.linspace(-5, 5, 100);
    y1 = sigmoid_d(x)
    y2 = tanh_d(x)

    plt.plot(x, y1, c='r', label="derivative of sigmoid")
    plt.plot(x, y2, c='b', label="derivative of tanh")
    plt.title("Derivatives of activation functions")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
