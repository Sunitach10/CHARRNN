

from dataloader import*
from train import*
from model import *
from dataloader import get_batches
# Define and print the net
#if __name__ == "__main__":
n_hidden=10
n_layers=2


net = CharRNN(chars, n_hidden, n_layers)
print(net)
if __name__ == "__main__":
# Declaring the hyperparameters
    batch_size = 32
    seq_length = 100
    n_epochs = 1 # start smaller if you are just testing initial behavior

# train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10000)

# Saving the model
    model_name = 'rnn_1_epoch.net'

    checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

    #with open(model_name, 'wb') as f:
    torch.save(net,'charmodel')
    
