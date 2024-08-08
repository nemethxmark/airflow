
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
        assert num_gpus == 0, 'No GPU support yet'
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
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
    



