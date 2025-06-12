import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network and allows
                      # us to access a lot of useful functions like:
                      # nn.Linear, nn.Embedding, nn.CrossEntropyLoss() etc.

from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
from torch.distributions.uniform import Uniform # So we can initialize our tensors with a uniform distribution
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

import lightning as L # lightning has tons of cool tools that make neural networks easier

import pandas as pd ## to create dataframes from graph input
import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.
                       
class WordEmbeddingWithEmbedding(L.LightningModule):

    def __init__(self):
        
        super().__init__()
        
        ## The first thing we do is set the seed for the random number generorator.
        ## This ensures that when someone creates a model from this class, that model
        ## will start off with the exact same random numbers as I started out with when
        ## I created this demo. At least, I hope that is what happens!!! :)
        L.seed_everything(seed=42)
        
        self.embed = nn.Embedding(4, 2) # 4 = number of words in the vocabulary, 2 = 2 numbers per embedding
        self.hidden_to_output = nn.Linear(2, 4, bias=False)
  
        ## We'll use CrossEntropyLoss in training_step()
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, input): 
          
        hidden = self.embed(input[0])    
        output_values = self.hidden_to_output(hidden)                
        
        return(output_values)
       
        
    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr=0.1)

    
    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i[0]) # run input through the neural network
        loss = self.loss(output_i, label_i[0]) ## self.loss = cross entropy
                    
        return loss
        
## Create the training data for the neural network.

## NOTE: nn.Embedding() applies one-hot-encoding to the input for us, so
## the data that we use for training will look different from before. 
inputsForEmbed = torch.tensor([[0], [1], [2], [3]]) ## NOTE: Troll2 = 0, is = 1, great = 2, Gymkata = 3
labels = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.], [0., 1., 0., 0.]])

datasetForEmbed = TensorDataset(inputsForEmbed, labels) 
dataloaderForEmbed = DataLoader(datasetForEmbed)

modelEmbed = WordEmbeddingWithEmbedding() 

print("Before optimization, the parameters are...")
for name, param in modelEmbed.named_parameters():
    print(name, param.data)
    
weights = modelEmbed.embed.weight.detach().numpy()
w1 = [weights[0][0], weights[1][0], weights[2][0], weights[3][0]]
print(w1)

w2 = [weights[0][1], weights[1][1], weights[2][1], weights[3][1]]
print(w2)

data = {
    "w1": w1,
    "w2": w2,
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")

## add the token each dot represents to the graph
## NOTE: For Troll2 and and Gymkata, we're adding offsets to where to print the tokens because otherwise
## they will be so close to each other that they will overlap and be unreadable.
plt.text(df.w1[0], df.w2[0], df.token[0], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # Troll 2
plt.text(df.w1[1], df.w2[1], df.token[1], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # is
plt.text(df.w1[2], df.w2[2], df.token[2], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # great
plt.text(df.w1[3], df.w2[3], df.token[3], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # Gymkata

plt.show()

trainer = L.Trainer(max_epochs=100)
trainer.fit(modelEmbed, train_dataloaders=dataloaderForEmbed)

print("After optimization, the parameters are...")
for name, param in modelEmbed.named_parameters():
    print(name, param.data)
    
weights = modelEmbed.embed.weight.detach().numpy()
w1 = [weights[0][0], weights[1][0], weights[2][0], weights[3][0]]
w2 = [weights[0][1], weights[1][1], weights[2][1], weights[3][1]]

data = {
    "w1": w1,
    "w2": w2,
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")

## add the token each dot represents to the graph
## NOTE: For Troll2 and and Gymkata, we're adding offsets to where to print the tokens because otherwise
## they will be so close to each other that they will overlap and be unreadable.
plt.text(df.w1[0]-0.2, df.w2[0]-0.3, df.token[0], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # Troll 2
plt.text(df.w1[1], df.w2[1], df.token[1], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # is
plt.text(df.w1[2], df.w2[2], df.token[2], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold') # great
plt.text(df.w1[3]-0.3, df.w2[3]+0.2, df.token[3], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold')# Gymkata

plt.show()

## Let's see what the model predicts
softmax = nn.Softmax(dim=0) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns

print(torch.round(softmax(modelEmbed(torch.tensor([0])).detach()), decimals=2)) ## print the predictions for "Troll2"
print(torch.round(softmax(modelEmbed(torch.tensor([1])).detach()), decimals=2)) ## print the predictions for "is"
print(torch.round(softmax(modelEmbed(torch.tensor([2])).detach()), decimals=2)) ## print the predictions for "great"
print(torch.round(softmax(modelEmbed(torch.tensor([3])).detach()), decimals=2)) ## print the predictions for "Gymkata"
