
$\qquad$ $\qquad$$\qquad$  **TDA 231 Machine Learning: Homework 4, part 2** <br />
$\qquad$ $\qquad$$\qquad$ **Goal: Fully-connected deep neural networks**<br />
$\qquad$ $\qquad$$\qquad$                   **Grader: Emilio, Simon** <br />
$\qquad$ $\qquad$$\qquad$                     **Due Date: 21/5** <br />
$\qquad$ $\qquad$$\qquad$                   **Submitted by: Adam Olsson, 19950418-xxxx, adaolss@student.chalmers.se** <br />

General guidelines:
* All solutions to theoretical and practical problems should be submitted in this Jupyter notebook.
* All discussion regarding practical problems, along with solutions and plots should be specified in this notebook. All plots/results should be visible such that the notebook do not have to be run. But the code in the notebook should reproduce the plots/results if we choose to do so.  
* Your name, personal number and email address should be specified above.

**Jupyter/IPython Notebook** is a collaborative Python web-based environment. This will be used in all our Homework Assignments. It is installed in the halls ES61-ES62, E-studio and MT9. You can also use google-colab: https://colab.research.google.com
to run these notebooks without having to download, install, or do anything on your own computer other than a browser.
Some useful resources:

1. https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/ (Quick-start guide)
2. https://www.kdnuggets.com/2016/04/top-10-ipython-nb-tutorials.html
3. http://data-blog.udacity.com/posts/2016/10/latex-primer/ (latex-primer)
4. http://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html (markdown)

In this assignment you will be using the `pytorch` package. Installation instructions can be found on the [pytorch homepage](https://pytorch.org/get-started/locally/). You don't need the GPU support.

# Theoretical problems

## [Calculating dimensions of output for CNN, 1 point]
### 1)
Assume you apply a convolutional layer with 8 3x3 filters (stride 1) on a rgb 28x13 image. What will the dimensions of the output be (assuming no padding is done in the convolution)?

Answer: 26x11x8

## [Counting parameters in a fully connected network, 1 point]
### 2)
Imagine you apply a two layer fully connected network to a 28x28 rgb image. The hidden layer has dimension 256 and the output is of size 10. How many parameters are necessary? Include the bias parameters.

Answer: 256x(28 * 28 * 3) + 256 + 10x256 + 10 = 604938

## [Counting parameters in a convolutional network, 1 point]
### 3)
Apply the following network to the same image, how many parameters are needed? Include bias parameters. Show your calculations.

* Convolutional layer with 8 3x3 filters (stride 1).

* Max pooling layer (2x2) (stride 2).

* Convolutional layer with 16 3x3 filter (stride 1).

* Fully connected layer to ouput of size 10.


Answer: 224+1168+19370= 20762



It is clear that the amount of parameters for a CNN can be a lot less than for a fully connected network.

## [Applying a filter, 2 points]

Image
$
\begin{bmatrix}2 & 2 & 1 & 2 \\
               -2 & -2 & -1 & 1 \\
               1 & 1 & 2 & 1 \\
               1 & 1 & 3 & 1 
\end{bmatrix}$
Filter 
$
\begin{bmatrix}1 & 1
\\-1 & -1
\end{bmatrix}$

Convolve the filter over the image and apply ReLU, use a stride of 2 with a bias of -2. Try to give an explanation for the output, what is the filter detecting?

Answer:$
\begin{bmatrix}
6 & 1 \\
0  & 0
\end{bmatrix}$

The filter is detecting a horizontal line or edge.

## [Gradients in CNN, 2 points]

Let us apply a 2x2 filter (with weights $W$) on a 2x$n$ greyscale image $x$ generating a 1x($n$-1) sequence $z$.

We then get 
$$ 
\text{Filter}: \, \left[ \begin{array}{ccc} 
W_{1,1} & W_{1,2}\\
W_{2,1} & W_{2,2}\\
\end{array}\right]
 \text{Image}: \, \left[ \begin{array}{ccc} 
x_{1,1} & x_{1,2}  & ... & x_{1,n}\\
x_{2,1} & x_{2,2}  & ... & x_{2,n} \\
\end{array}\right]
 \text{Output}: \, \left[ \begin{array}{ccc} 
z_{1} & z_{2}  & ... & z_{n-1}\\
\end{array}\right]
$$

### 4, a)
Write $z_i$ as a function of $W$ and $x$.
$$
\begin{align}
z_i = W_{1,1}x_{1,i} + b_{1,1} + W_{2,1}x_{2,i} + b_{2,1} + W_{1,2}x_{1,i+1} + b_{1,2} + W_{2,1}x_{2,i+1} + b_{2,2} \\
\end{align}
$$

### 4, b)
Lets assume we know $\frac{dE}{dz_i}$. Calculate the gradient of the error with respect to the filter and the image. You can ignore the edge cases by assuming $x_{i,j}=0$ for i outside the image and similary for $z_i$. 
\begin{align}
\frac{\partial E}{\partial W_{i,j}} = \sum_{k}^{n-1}\frac{\partial E}{\partial z_{k}}\frac{\partial z_{k}}{W_{i,j}} = \sum_{k}^{n-1}\frac{\partial E}{\partial z_{k}}x_{k,(i+(j-1))}\\
\frac{dE}{dx_{i,j}}= \sum_{k}^{n-1}\frac{\partial E}{\partial z_{k}}\frac{\partial z_{k}}{x_{i,j}} = \frac{\partial E}{z_{j-l}}\frac{\partial z_{j-1}}{x_{i,j}} + \frac{\partial E}{z_{j}}\frac{\partial z_{j}}{x_{i,j}} = \frac{\partial E}{z_{j-l}}W_{i,2} + \frac{\partial E}{z_{j}}W_{i,1}\\
\end{align}

# Practical problems

## [Building a CNN with pytorch, 3 points]

### 5)
Build a convolutional network from the following specification:
* Input is a single channel 28x28 image (black/white).
* Conv1  has 8 3x3 filters with ReLU activation (stride 1)
* Max pooling (2x2) layer (stride 2)
* Conv2  has 16 3x3 filters with ReLU activation (stride 1)
* Reshape data for fully connected layer
* Three fully connected layers that will have dimensions 32 and 64 and then 10 for output.
    * The first two layers have ReLU activation while the final layer doesn't have any.

You will have to figure out some of the implicit dimensions of the different layers and all necessary methods are already imported. You can use tensor.view() to change shape of a tensor. 


```python
from torch.nn import Conv2d, Linear
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax



class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = Conv2d(1,8,(3,3))
        self.conv2 = Conv2d(8,16,(3,3))
        
        self.fc1   = Linear(16*11*11, 32)
        self.fc2   = Linear(32,64)
        self.fc3   = Linear(64,10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x,2)
        x = relu(self.conv2(x))
        x = x.view(-1,11*11*16)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        
        out = self.fc3(x)
        return log_softmax(out, dim=1)
```


```python
#Import data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
training_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)

n = len(training_data)
n_train = int(0.9 * n)
n_val = n - n_train
training_data, validation_data = random_split(training_data, [n_train, n_val])
training_loader   = DataLoader(training_data, batch_size = 64, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size = 64)

```


```python
#Add optimizer and initialize model
import torch
model = MyConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```


```python
#Train network:
import torch.nn.functional as F
epochs = 2
for epoch in range(epochs):
    for (data, target) in training_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```

### How well does the network perform?


```python
def validation_error(model):
    """
    Compute the validation error of the given model in percent.
    """
    error = 0.0
    for (data, target) in validation_loader:
        error += torch.sum(model(data).argmax(dim = 1) != target).float()
    error /= float(len(validation_data))
    return 100.0 * error
    
print("Validation error: {0:.2f}%.".format(validation_error(model)))
```

    Validation error: 1.83%.


### [A better classifier, 2 points]
Compare with http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354

Make changes to get a validation error lower than 1.5%. 
you are allowed (and even encouraged) to use all other available features
from the `pytorch` library.
Helpful `pytorch` sub-modules are the [nn](https://pytorch.org/docs/stable/nn.html) module and the [optmizers](https://pytorch.org/docs/stable/optim.html) module. 

Some suggestions on things to try out are:
* size of layers
* number of layers
* learning rate
* different optimizer
* number of epochs

Feel free to be creative

### Changes
* Using filter of size 5x5 insted of 3x3
* One extra layer of 64 noeds
* increased epocs to 3


```python
from torch.nn import Conv2d, Linear
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax

class MyImprovedConvNet(nn.Module):
    def __init__(self):
        super(MyImprovedConvNet, self).__init__()
        self.conv1 = Conv2d(1,8,(5,5))
        self.conv2 = Conv2d(8,16,(5,5))
        
        self.fc1   = Linear(16*8*8, 32)
        self.fc2   = Linear(32,64)
        self.fc2_1 = Linear(64,64)
        self.fc3   = Linear(64,10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x,2)
        x = relu(self.conv2(x))
        x = x.view(-1,8*8*16)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc2_1(x))
        
        out = self.fc3(x)
        return log_softmax(out, dim=1)
```


```python
#Initialize
import torch
model = MyImprovedConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```


```python
#Train:
import torch.nn.functional as F
epochs = 4
for epoch in range(epochs):
    for (data, target) in training_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

```


```python
print("Validation error: {0:.2f}%.".format(validation_error(model)))
```

    Validation error: 1.22%.

