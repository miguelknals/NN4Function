# NN4Function

The purpose of this repository is to play with a multidimensional function
and approximate it with a (simple) Neural Network

##  MyFunctionData class (C01_function_data.py)

This class defines de function data. You need to define Xi randge and the function itself defined in multif.
```
    def multif(self, x):
        f1= math.sin (x[0])
        f2= math.sin (x[1]*2)/2
        f3= math.sin (x[2]*4)/4
        f= (f1+f2+f3) * x[3]
        return f
```
## MyDataset class (C02_datset.py)

We define the custom dataset with the `__init__` . Data is read from the text file X data and F(X) is stored in `data` and `labels` tensors 

Also we define `__len__` and `__getitem__` . 

## MyModel class (C03_model.py) 

We define a simple neural network, you can easily customize to play.

class MyModel(nn.Module):
```
    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.fc0 = nn.Linear(n_features, n_hidden)
        self.act0 = nn.Sigmoid() # nn.ReLU() # 
        
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.act1 = nn.Sigmoid() # nn.ReLU() # 
        
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 =  nn.Sigmoid() # nn.ReLU() # 
        
        self.fc3 = nn.Linear(n_hidden, n_outputs)
        
        
    
    def forward(self, x):
        x = self.act0(self.fc0(x))
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x=self.fc3(x)
        
        return x
```
## many.py
















(C) 2022 miguel canals (http://www.mknals.com) MIT License 