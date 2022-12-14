#import argparse
#from ast import MatchValue
#from symbol import testlist_comp

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from C02_dataset import MyDataset
from C03_model import MyModel
from C01_function_data import MyFunctionData

# reproducibility
torch.manual_seed(0)
import random
random.seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device {}".format(device))

def train_single_epoch(epoch, train_dataloader, network, 
                       optimizer, criterion, device):
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
           
    return avg_loss



def eval_single_epoch(epoch, verification_dataloader, network, criterion, 
                       device,label):
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
                
    return verification_loss




def minimum(a, b):
    if a <= b:
        return a
    else:
        return b

def train_model(config):
    
    # values
    filename=config["dataset_filename"]
    nvalues=config["n_values"]
    batch_size=config["batch_size"]
    n_features=config["n_features"]
    n_hidden=config["n_hidden"]
    n_outputs=config["n_outputs"]
    lr=config["learning_rate"]
    epochs=config["epochs"]
    
    # we generate dataset from the MyFunctionData class 
    # that will generate a filenmae with the data
    MyFunctionData(filename,nvalues) # temporary filename and number of points
    
    # we create the datset based on the previous filename
    my_dataset = MyDataset(filename=filename) # my dataset using the custom dataset

    len_my_dataset= my_dataset.__len__() 
    l_validation=minimum(int(0.1*nvalues),1000) # val lenght 10% of data, max 1000.
    l_test=minimum(int(0.1*nvalues),1000)       # test lenght 10% of data, max 1000.
    l_train=len_my_dataset-l_validation-l_test  # train data the difference
    train_dataset, validation_dataset , test_dataset \
        = random_split( my_dataset, [l_train, l_validation, l_test]) # we split dataset
        
    # now create the dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader =DataLoader(test_dataset, batch_size=batch_size)     

    # Model creation and loss and optimization criteria
    network = MyModel(n_features, n_hidden, n_outputs).to(device) # we create the model
    criterion = nn.MSELoss() # loss function is mean squared error (squared L2 norm) 
    #optimizer = optim.SGD(network.parameters(), lr=lr) # Stochastic Gradien Descent
    optimizer = optim.Adam(network.parameters(), lr=lr) # Adam in general works better.
    #print (network)
    

    tr_losses = []; va_losses = []; te_losses=[] # we initialize loss info

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss= train_single_epoch(epoch, train_dataloader, network, optimizer, criterion, device )
        # there is no difference between validation and test in this program
        # as validation set is not used to modify trained model
        # see https://datascience.stackexchange.com/questions/88864/how-is-the-validation-set-processed-in-pytorch
        #
        # The validation test is used instead of the test set when you’re doing hyperparameter tuning.
        # Evaluating the set in itself does not adjust anything (see torch.no_grad()), it just gives you
        # the chance to compute some metrics (let’s say accuracy and so on), tweak hyperparameters manually
        # or with ray, evaluate the validation set over and over again and see how metrics are affected.
        #
        # In this example we do not tune any hyperparameter tunning, we just keep a framewor.keys()
        
        validation_loss = eval_single_epoch(epoch, validation_dataloader, network, criterion, device,"Validation" )
        test_loss = eval_single_epoch(epoch, test_dataloader, network, criterion, device, "Test      " )    
        tr_losses.append(avg_loss); va_losses.append(validation_loss); te_losses.append(test_loss)
        
        
        
    display_model_results(tr_losses, va_losses, te_losses ) # display losses in a simple diagram


     
    return network
        

def display_model_results(tr_losses, va_losses, te_losses):
    epochs= len(tr_losses)
    plt.xlabel('Epoch')
    plt.ylabel('NLLLoss')
    plt.plot(tr_losses, label='train')
    plt.plot(va_losses, label='validation')
    plt.plot(te_losses, label='test')
    plt.legend(loc="upper right")
    plt.show()
    #plt.savefig("dummy_name.png")
    return



if __name__ == "__main__":
    

    # Config dictionary will contain model variables
    config = {
        "dataset_filename": "tmp.txt", # file to store function dataset
        "n_values": 5000, # number of points to be generated on the dataset
        "epochs":50,           # for training
        "learning_rate": 1e-2, # for training
        "n_features":4,        # number of input features
        "n_hidden":10,         # number of hidden nodes in intermediate layers
        "n_outputs":1,         # number of output features 
        "batch_size": 64        
    }
    
    
    Net= train_model(config) # we pass config values to create the neuronal network
    print (Net) # print NN model
    
    for layer in Net.children(): # print layer data
        if isinstance(layer, nn.Linear):
            print("--------------------------")
            print(layer.state_dict()['weight'])
            print(layer.state_dict()['bias'])
            
    print("EOF") 