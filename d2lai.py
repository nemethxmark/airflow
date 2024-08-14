
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# UTILS ------------------------------------------------------------
def add_to_class(Class):
    """Allows us to register functions as methods in a class after the class has been created."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper

class HyperParameters:
    """Saves all arguments in a class's __init__ methods as class attributes. This allows us to extend constructor call signatures implicitly without additional code."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplementedError

class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

B = B(1, 2, 3)

class ProgressBoard(d2l.HyperParameters):
    """Plot experiment progress interactively while it is going on"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'], fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplementedError

board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)

class Module(nn.Module, d2l.HyperParameters):  # @save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches + self.trainer.epoch
            n = self.trainer.num_train_batches / self.trainer.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.trainer.plot_valid_per_epoch
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(*batch[:-1], batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(*batch[:-1], batch[-1])
        self.plot('loss', l, train=False)
        return l

    def configure_optimizers(self):
        raise NotImplementedError



# DATA --------------------------------------------------------------------------- 
'''The DataModule class is the base class for data. Quite frequently the __init__ method
is used to prepare the data. This includes downloading and preprocessing if needed. 
The train_dataloader returns the data loader for the training dataset. A data loader
is a (Python) generator that yields a data batch each time it is used. This batch is 
then fed into the training_step method of Module to compute the loss'''
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()
    def get_dataloader(self, train):
        raise NotImplementedError
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)
        













      
# TRAINER --------------------------------------------------------------------------- 
  
class Trainer(d2l.HyperParameters):  #@save
    ''' The Trainer class trains the learnable parameters in the Module class 
    with data specified in DataModule. The key method is fit, which accepts two 
    arguments: model, an instance of Module, and data, an instance of DataModule. 
    It then iterates over the entire dataset max_epochs times to train the model. 
    As before, we will defer the implementation of this method to later chapters'''
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    def fit_epoch(self):
        raise NotImplementedError
    def prepare_batch(self, batch):
        def prepare_batch(self, batch):
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]
        return batch
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1      




class SyntheticRegressionData(d2l.DataModule):  #@save
    """We generate each label by applying a ground truth linear function, 
    corrupting them via additive noise, drawn independently and identically for each example"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
        
    def get_dataloader(self, train):
        '''Training machine learning models often requires multiple passes over a dataset, 
        grabbing one minibatch of examples at a time. This data is then used to update the model.
        It takes a batch size, a matrix of features, and a vector of labels, and generates minibatches 
        of size batch_size. As such, each minibatch consists of a tuple of features and labels. 
        Note that we need to be mindful of whether we’re in training or validation mode: in the former, 
        we will want to read the data in random order, whereas for the latter, 
        being able to read data in a pre-defined order may be important for debugging purposes.
        
        Y = Xw + b 
        Y (nxm) - n examples, m output features
        X (nxk) - n examples, k input features
        w (kxm) - k input features, m output features
        b (1xm) - 1 bias, m output features
        
        Generates a data loader that yields batches of data for either training or validation. 
        For training, it shuffles the data to ensure randomness, while for validation, it maintains the original order.
        
        How it works:
        1. If the 'train' parameter is True, it means we are preparing the data for training. 
        In this case, we create a list of indices from 0 to the number of training examples (self.num_train) 
        and shuffle them to ensure that each training epoch uses a different order of examples.
        2. If the 'train' parameter is False, we are preparing the data for validation. 
        We create a list of indices from the number of training examples to the total number of examples 
        (training examples + validation examples). For validation, we don't need to shuffle the indices.
        3. We then iterate over the indices list, taking 'self.batch_size' number of indices at a time. 
        These indices are used to select a batch of examples from the feature matrix (self.X) and the labels vector
        (self.y).
        4. Each batch of features and labels is then yielded. When this function is used in a for loop, 
        it will yield a new batch in each iteration until all examples have been used.
        if train:
            indices = list(range(0, self.num_train))
            # The examples are read in random order
            random.shuffle(indices)
        else:
            indices = list(range(self.num_train, self.num_train+self.num_val))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = torch.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
            
            
            
        Data loaders are a convenient way of abstracting out the process of loading and manipulating data. 
        This way the same machine learning algorithm is capable of processing many different types and 
        sources of data without the need for modification. One of the nice things about data loaders 
        is that they can be composed
        '''
            
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
            
        
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
print("X:", data.X)
print("y:", data.y)
print('features:', data.X[0],'\nlabel:', data.y[0])
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)

'''A data loader in PyTorch is a utility that helps with loading and preprocessing data. It provides an efficient way to iterate
over a dataset in small manageable batches, rather than loading the entire dataset into memory at once. This is especially 
useful when working with large datasets that cannot fit into memory.
In the `SyntheticRegressionData` class you provided, the `get_dataloader` method is used to create a data loader that yields
batches of synthetic regression data for either training or validation.
Here's a step-by-step explanation of what the `get_dataloader` method does:
When you iterate over the data loader in your training or validation loop, it will yield batches of features and
labels. Each batch is a tuple containing a feature matrix and a labels vector. The feature matrix has shape 
`(batch_size, num_features)`, and the labels vector has shape `(batch_size, 1)`.'''


class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
        
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    
    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)






# ONE LAYER, predefined stuff  --------------------------------------------------------------------------- 
    
    
                
class LinearRegression(d2l.Module):  
    """
    The linear regression model implemented with high-level APIs.
    This class represents a single-layer network, also known as a fully connected
    or dense layer. Each input is connected to each output by means of a 
    matrix-vector multiplication. This implementation uses predefined layers 
    from a framework, allowing us to focus on the architecture of the model 
    rather than the details of the operations.
    """
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        
    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        """
        The MSELoss class computes the mean squared error 
        returns the average loss over examples. It is faster 
        (and easier to use) than implementing our own.
        """
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        """ 
        Minibatch SGD is a standard tool for optimizing 
        neural networks and thus PyTorch supports it alongside 
        a number of variations on this algorithm in the optim module.
        When we instantiate an SGD instance, we specify the 
        parameters to optimize over, obtainable from our 
        model via self.parameters(), and the learning
        rate (self.lr) required by our optimization algorithm. 
        
        expressing our model through high-level APIs of a deep learning
        framework requires fewer lines of code. We did not have to allocate 
        parameters individually, define our loss function, or implement minibatch SGD
        
        The training loop itself is the same as the one we implemented from scratch. 
        So we just call the fit method
        """
        
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)
    
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)  
w, b = model.get_w_b() 
print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')





# Generalization --------------------------------------------------------------------------- 
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
    
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
def l2_penalty(w):
    return (w ** 2).sum() / 2
def train_scratch(lambd):
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
      
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)






        
class WeightDecay(d2l.LinearRegression):
    
    """ Yes, the code snippet you provided does apply L2 regularization, also known as weight decay.
        In the `configure_optimizers` method, the `torch.optim.SGD` optimizer is being configured with a
        `weight_decay` parameter. This `weight_decay` parameter applies L2 regularization to the model
        parameters during optimization.
        L2 regularization encourages the weights to 
        be small, which can help prevent overfitting by discouraging 
        the model from relying too heavily on any single feature. 
        The `weight_decay` parameter controls the strength of this regularization:
        a larger value means stronger regularization. """
        
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
        
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))






''' Y = softmax (Xw + b)
where
Y (nxm) - n examples, m probabilities of each category (sum is 1)
X (nxk) - n examples, k input features
w (kxm) - k input features, m output features (or categories)
b (1xm) - 1 bias, m output features (or categories)
Softmax  - provide a vector of probabilities for each example. Sum of praobabilites is 1. Used for multi class classification.
Loss function is Log likelyhood - SUM(SUM(y * log(y_hat))) - where y is the real proabbaility of a 
class and y_hat is the predicted probability of it. Sum all the classes and all the examples.
'''
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import time
import torchvision
from torchvision import transforms
d2l.use_svg_display()
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset.
        Fashion-MNIST consists of images from 10 categories, each represented by 6000 
        images in the training dataset and by 1000 in the test dataset. A test dataset 
        is used for evaluating model performance (it must not be used for training).
        Consequently the training set and the test set contain 60,000 and 10,000 images, respectively.
        Each image consists of a 28x28=784 grid of grayscale pixel values - 784 input features and 10 classes
        Layers - 2,4,8,16,32,64,128"""
    
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                        num_workers=self.num_workers)
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
        """Plot a list of images."""
        raise NotImplementedError
    
    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
data = FashionMNIST(resize=(32, 32))
batch = next(iter(data.val_dataloader()))
data.visualize(batch)







# One Layer Classification (SoftMax Regression) --------------------------------------------------------------------------- 
                 
class Classifier(d2l.Module):  #the model
    """The base class of classification models. In the validation_step we report both the loss 
    value and the classification accuracy on a validation batch. We draw an update
    for every num_val_batches batches. This has the benefit of generating the
    averaged loss and accuracy on the whole validation data. These average
    numbers are not exactly correct if the final batch contains fewer examples,
    but we ignore this minor difference to keep the code simple."""
    
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
        It is a matrix of 0s and 1s. 1 if the prediction is correct, 0 otherwise.
        Y_hat is the predicted probability of each class, rows represent examples, columns represent classes.
        
        Tracing is a form of logging that is particularly useful for understanding the 
        flow of execution and performance characteristics of your application.
        It's often used in distributed systems to understand how a single request
        flows through various services, but it can also be useful in a single 
        service to understand how a request is handled logger.debug (Entering/Exiting a method) by logger.info()
        make it possible to turn that on/off runtime to reduce the number of logs
        """
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
    def parameters(self):
        return [self.W, self.b]
    
    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)
    
    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()
      
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
# data = d2l.FashionMNIST(batch_size=256)
# model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
# trainer = d2l.Trainer(max_epochs=10)
# trainer.fit(model, data)
# X, y = next(iter(data.val_dataloader()))
# preds = model(X).argmax(axis=1)
# preds.shape
# wrong = preds.type(y.dtype) != y
# X, y, preds = X[wrong], y[wrong], preds[wrong]
# labels = [a+'\n'+b for a, b in zip(
#     data.text_labels(y), data.text_labels(preds))]
# data.visualize([X, y], labels=labels)    
    
    
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        ''' Y = Xw + b calulations'''
        return self.net(X)
    
    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    
# data = d2l.FashionMNIST(batch_size=256)
# model = SoftmaxRegression(num_outputs=10, lr=0.1)
# trainer = d2l.Trainer(max_epochs=10)
# trainer.fit(model, data)







"""
This script demonstrates the use of a MultiLayer Perceptron (MLP) which can catch non-linearities with hidden layers.
The MLP is a type of deep neural network with fully connected neurons. Like concatenating multiple linear models
The MLP works in the following way:
H1 = relu(Xw1 + b1)
H2 = relu(H1w2 + b2)
This script also plots the activation functions - ReLU, Sigmoid, and Tanh.
- **ReLU (Rectified Linear Unit)**: ReLU is beneficial due to its simplicity and efficiency, 
helping to mitigate the vanishing gradient problem in deep networks, and it's commonly used 
in convolutional neural networks (CNNs) and deep learning models.
- **Sigmoid**: The Sigmoid function is beneficial because it maps any input into a range 
between 0 and 1, making it useful for binary classification problems, and it's often used 
in the output layer of binary classification models.
- **Tanh**: The Tanh function, like the sigmoid function, helps to mitigate the vanishing 
gradient problem but maps inputs to a range between -1 and 1, making it useful when negative 
outputs are meaningful; it's often used in recurrent neural networks (RNNs).
"""
import torch
import matplotlib.pyplot as plt
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# Compute the y values for each function
relu_y = torch.relu(x)
sigmoid_y = torch.sigmoid(x)
tanh_y = torch.tanh(x)
# Create the plot
plt.figure(figsize=(5, 2.5))
# Plot each function
plt.plot(x.detach(), relu_y.detach(), label='relu(x)')
plt.plot(x.detach(), sigmoid_y.detach(), label='sigmoid(x)')
plt.plot(x.detach(), tanh_y.detach(), label='tanh(x)')
# Set the y-axis limits
plt.ylim([-1, 3])
# Add a legend
plt.legend()
# Show the plot
plt.show()

class MLPScratch(d2l.Classifier):
    '''use nn.Parameter to automatically register a class attribute as a parameter to be tracked by autograd'''
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2
# model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
# data = d2l.FashionMNIST(batch_size=256)
# trainer = d2l.Trainer(max_epochs=10)
# trainer.fit(model, data)
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
        
# model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
# trainer.fit(model, data)




def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
    
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
        
model = DropoutMLP(**hparams)
trainer.fit(model, data)



'''Alongside giant datasets and powerful hardware, great software tools have played an indispensable role in the rapid progress of deep learning. Starting with the pathbreaking Theano library released in 2007, flexible open-source tools have enabled researchers to rapidly prototype models, avoiding repetitive work when recycling standard components while still maintaining the ability to make low-level modifications. Over time, deep learning’s libraries have evolved to offer increasingly coarse abstractions. Just as semiconductor designers went from specifying transistors to logical circuits to writing code, neural networks researchers have moved from thinking about the behavior of individual artificial neurons to conceiving of networks in terms of whole layers, and now often design architectures with far coarser blocks in mind.

To implement these complex networks, we introduce the concept of a neural network module. A module could describe a single layer, a component consisting of multiple layers, or the entire model itself! One benefit of working with the module abstraction is that they can be combined into larger artifacts, often recursively.'''

import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
X = torch.rand(2, 20)
net(X).shape



class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
		# Note that it takes X as input, calculates the hidden representation with the activation function applied, and outputs its logits. In this MLP implementation, both layers are instance variables
		
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)
# Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


'''To implement these complex networks, we introduce the concept of a neural network module. 
A module could describe a single layer, a component consisting of multiple layers, or
the entire model itself! One benefit of working with the module abstraction is that they can be combined into larger artifacts, often recursively.
module is represented by a class. Any subclass of it must define a forward propagation method that transforms
its input into output and must store any necessary parameters. Note that some modules do not require any parameters 
at all. Finally a module must possess a backpropagation method, for purposes of calculating gradients. '''
class MLP(nn.Module):
    def __init__(self):
        '''Note that it takes X as input, calculates the hidden 
        representation with the activation function applied,
        and outputs its logits. In this MLP implementation, 
        both layers are instance variables. To see why this is reasonable, 
        imagine instantiating two MLPs, net1 and net2, and training
        them on different data. Naturally, we would expect them 
        to represent two different learned models.
        '''
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)
    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
import torch
from torch import nn
from torch.nn import functional as F
'''In short, nn.Sequential defines a special kind of Module,
the class that presents a module in PyTorch. It maintains
an ordered list of constituent Modules. Note that each of
the two fully connected layers is an instance of the Linear
class which is itself a subclass of Module. The forward propagation
(forward) method is also remarkably simple: it chains each module
in the list together, passing the output of each as input to the
next. Note that until now, we have been invoking our models via 
the construction net(X) to obtain their outputs
LazyLinear - Pythorch function for layer, input features number is coming from the batch it gets, the output is passed as a param'''
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
X = torch.rand(2, 20)
net(X).shape
    
    
class MySequential(nn.Module):
    '''Designed to daisy-chain other modules together. To build our own simplified MySequential, we just need to define two key methods:
    A method for appending modules one by one to a list.
    A forward propagation method for passing an input through the chain of modules, in the same order as they were appended.'''
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
    
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
net[2].state_dict() # 2-es model, parameterei (bias and weights)
type(net[2].bias), net[2].bias.data # 2-es model, bias
[(name, param.shape) for name, param in net.named_parameters()] # all parameters at once

class FixedHiddenMLP(nn.Module):
    '''In this model, we implement a hidden layer whose weights (self.rand_weight)
    are initialized randomly at instantiation and are thereafter constant. 
    This weight is not a model parameter and thus it is never updated by
    backpropagation. The network then passes the output of this “fixed” layer through a fully connected layer.  
    Note that before returning the output, our model did something unusual.
    We ran a while-loop, testing on the condition its  norm is larger than ,
    and dividing our output vector by  until it satisfied the condition. 
    Finally, we returned the sum of the entries in X. To our knowledge, 
    no standard neural network performs this operation. Note that this 
    particular operation may not be useful in any real-world task.
    Our point is only to show you how to integrate arbitrary code 
    into the flow of your neural network computations.
    '''
    
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
'''We can mix and match various ways of assembling modules together. In the following example, we nest modules in some creative ways.'''
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)
    def forward(self, X):
        return self.linear(self.net(X))
chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)

'''Individual layers can be modules. Many layers can comprise a module. Many modules can comprise a module.
 A module can contain code. Modules take care of lots of housekeeping, including parameter initialization and
 backpropagation. Sequential concatenations of layers and modules are handled by the Sequential module.'''
 
 
 
 '''We can initialize parameters using built-in and custom initializers.'''
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
 def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5
net.apply(my_init)
net[0].weight[:2]



# Torch save and load 
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
x2
# Model save and load 
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# GPU
'''We can specify devices, such as CPUs and GPUs, 
for storage and calculation. By default, tensors
are created in the main memory and then the CPU is used for calculations.
Be noted that the cpu device means all physical CPUs and memory. 
This means that PyTorch’s calculations will try to use all CPU cores.
However, a gpu device only represents one card and the corresponding memory'''
def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')
def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')
def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]
'''By default, tensors are created on the CPU. We can query the device where the tensor is located.'''
x = torch.tensor([1, 2, 3])
x.device
'''It is important to note that whenever we want to operate on multiple terms, 
they need to be on the same device. For instance, if we sum two tensors, 
we need to make sure that both arguments live on the same device—otherwise
the framework would not know where to store the result or even how to
decide where to perform the computation.'''
X = torch.ones(2, 3, device=try_gpu())
X
'''we can specify a storage device when creating a tensor. Next, we create the tensor variable X on the first gpu'''
'''Copying to the same device , it is slow an bad practice, keep them on the same'''
Z = X.cuda(1)
print(X)
print(Z)
'''when we print tensors or convert tensors to the NumPy format, 
if the data is not in the main memory, the framework will copy it
to the main memory first, resulting in additional transmission overhead.
Even worse, it is now subject to the dreaded global interpreter lock 
that makes everything wait for Python to complete.'''

net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
net(X)
net[0].weight.data.device




    



