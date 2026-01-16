import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

print(70*"-")
print('"What I cannot create, I do not understand" (Richard Feynman)')
print(70*"-")
##############

mnist = fetch_openml(name = 'mnist_784')
print(mnist.keys())

data = mnist.data
label = mnist.target

np.random.seed(42)

n = np.random.choice(np.arange(data.shape[0]+1))

print(n)

test_img = data.iloc[n].values
test_label = mnist.target.iloc[n]

print(test_img.shape)

side = int(np.sqrt(test_img.shape[0]))
reshaped_img = test_img.reshape(side,side) 

print(f"image label:{test_label}")

plt.imshow(reshaped_img)
plt.show()

w1 = np.ones((784,4))*0.01
z1 = np.dot(data,w1)

print(z1.shape)

w2 = np.ones((4,10))
z2 = np.dot(z1,w2)

#activation functions for hidden layer

def sigmoid(z:np.ndarray) -> np.ndarray:
    return (1/(1+np.exp(-z)))

def relu(z:np.ndarray) -> np.ndarray:
    return np.maximum(0,z)

def tanh(z:np.ndarray) -> np.ndarray:
    return np.tanh(z)

def leaky_relu(z:np.ndarray)->np.ndarray:
    return np.where(z>0,z,z*0.01)

# output layer will be softmax function
# since this is a multiclass classification problem
def softmax(z:np.ndarray) -> np.ndarray:
    e = np.exp(z-np.max(z))
    return e/np.sum(e,axis=0)

# normalize function to scale the inputs to the [0, 1]
# we will use min-max scaling in this implementation

def normalize(x : np.ndarray) -> np.ndarray:
    return (x-np.min(x))/(np.max(x)-np.min(x))

#one-hot-encode function, which will turn the array of labels from an n-sized vector 
# to an n x m array (where m is the number of possible outputs).

def one_hot_encode(x: np.ndarray, n_labels: int) -> np.ndarray:
    return np.eye(n_labels)[x]

# derivative for calculating the gradient descent

def derivative(func_name: str, z: np.ndarray) -> np.ndarray:
    if func_name == "sigmoid":
        return sigmoid(z) * (1 - sigmoid(z))
    elif func_name == 'relu':
        y = (z > 0) * 1
        return y
    elif func_name =='tanhl':
        return 1 - np.square(tanh(z))
    elif func_name == 'leaky_relu':
        return np.where(z > 0, 1, 0.01)
    return "No such activation"

class Neural_Network:
    
    def __init__(self,X: np.ndarray,Y:np.ndarray,x_test: np.ndarray,
                 y_test: np.ndarray,activation:str,num_labels:int,
                 architecture:list[int]) -> None:
        
        self.X = normalize(X) #normalize the data with max-min between 0 and 1
        
        assert np.all((self.X>=0) | (self.X<=1))
        
        self.X, self.X_test = X.copy(), x_test.copy()
        self.y, self.y_test = Y.copy(), y_test.copy()
        self.layers = {} 
        self.activation = activation
        self.architecture = architecture
        
        assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]
        
        self.parameters = {}
        self.num_labels = num_labels
        self.m = X.shape[1]
        self.architecture.append(self.num_labels)
        self.num_input_features = X.shape[0]
        self.architecture.insert(0, self.num_input_features)
        self.L = len(architecture) 
        
        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)

def initialize_parameters(self)->None:
    for i in range(1,self.L):
        print(f"Initializing parameters for layer: {i}.")
        self.parameters["w"+str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01
        self.parameters["b"+str(i)] = np.zeros((self.architecture[i], 1))
    
def forward(self):
    
    params = self.parameters
    self.layers["a0"] = self.X
    
    for l in range(1,self.L-1):
            self.layers["z" + str(l)] = np.dot(params["w" + str(l)],self.layers["a"+str(l-1)]) + params["b"+str(l)]
            self.layers["a" + str(l)] = eval(self.activation)(self.layers["z"+str(l)])
            assert self.layers["a"+str(l)].shape == (self.architecture[l], self.m)
    
    self.layers["z" + str(self.L-1)] = np.dot(params["w" + str(self.L-1)],
               self.layers["a"+str(self.L-2)]) + params["b"+str(self.L-1)]
    
    
    self.layers["a"+str(self.L-1)] = softmax(self.layers["z"+str(self.L-1)])
    
    self.output = self.layers["a"+str(self.L-1)]
    
    assert self.output.shape == (self.num_labels, self.m)
    assert all([s for s in np.sum(self.output, axis=1)])
    
    cost = - np.sum(self.y * np.log(self.output + 0.000000001))
    
    return cost, self.layers

def backpropagation(self):
     
    derivatives = {}
    dZ  = self.output - self.y
    assert dZ.shape == (self.num_labels,self.m)
     
    dW = np.dot(dZ, self.layers['a'+self.L-2].T)/self.m
    db = np.sum(dZ,axis=1,keepdims=True)/self.m
     
    dA_prev = np.dot(self.parameters['w'+self.L-1].T,dZ)
     
    derivative['dW'+str(self.L-1)] = dW
    derivative['dB'+str(self.L-1)] = db
     
    for i in range(self.L-2,0,-1):
        dZ = dA_prev*derivative(self.activation,self.layers['z'+str(i)])
        dW = i/self.m*np.dot(dZ, self.layers['z'+ str(i-1)].T)
        db = i/self.m*np.sum(dZ,axis=1,keepdims=True)
        if i>1:
            dA_prev = np.dot(self.parameters['w'+str(i)].T,(dZ))
        derivatives["dW" + str(l)] = dW
        derivatives["db" + str(l)] = db
        self.derivatives = derivatives
        
    return self.derivatives


     
     