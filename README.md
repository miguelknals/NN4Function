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
## main.py

We call a 'train_model(config)' with the hyperparmeters


### 'train_model(config)'

```
# we generate dataset from the MyFunctionData class 
# that will generate a filenmae with the data
MyFunctionData(filename,nvalues) # temporary filename and number of points

# we create the datset based on the previous filename
my_dataset = MyDataset(filename=filename) # my dataset using the custom dataset
```

`my_dataset` is splited for train, validation and test. With 
these dataset we create te dataloaders. 

```
# Model creation and loss and optimization criteria
network = MyModel(n_features, n_hidden, n_outputs).to(device) # we create the model
criterion = nn.MSELoss() # loss function is mean squared error (squared optimizer = optim.Adam(network.parameters(), lr=lr) # Adam in general works better.
```

Then the main loop:

```
for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss= train_single_epoch(epoch, train_dataloader, network, optimizer, criterion, device )
        
        validation_loss = eval_single_epoch(epoch, validation_dataloader, network, criterion, device,"Validation" )
        test_loss = eval_single_epoch(epoch, test_dataloader, network, criterion, device, "Test      " )    
        tr_losses.append(avg_loss); va_losses.append(validation_loss); te_losses.append(test_loss)
```

##### `train_single_epoch`

Main loop. Evaluates batches, loss and logs the lossess.

```
    network.train() # train=True for inside de model
    avg_loss= None
    avg_weight=0.1
    for batch_idx, (data, target) in enumerate(train_dataloader ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output= network(data)
        loss= criterion(output, target) # es la dif cuadratica media.
        loss.backward()
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_dataloader.dataset),
              100. * batch_idx / len(train_dataloader), avg_loss))
    
    # print last Avg loss           
    print('Train Epoch: {} \tLast loss: {:.6f} Last avg loss: {:.6f}'.format(
            epoch, loss.item(), avg_loss))
```

##### `eval_single_epoch`

Similar to `train_single_epoch` but just evaluation

```
    network.eval()
    verification_loss=0
    avg_loss= None
    avg_weight=0.1
    with torch.no_grad():
        for data, target in verification_dataloader:
            data, target = data.to(device), target.to(device)
            output = network(data)        
            verification_loss= criterion(output, target) # es la dif cuadratica media.
            if avg_loss:
                avg_loss = avg_weight * verification_loss.item() + (1 - avg_weight) * avg_loss
            else:
                avg_loss = verification_loss.item()
                
        print('{}: Average loss: {:.4f}'.format(label, 
            verification_loss ))
```


   


















(C) 2022 miguel canals (http://www.mknals.com) MIT License 