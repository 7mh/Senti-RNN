import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from string import punctuation
from collections import Counter
from tqdm import tqdm
import pdb

test_x = np.load('./data/test_x.npy')
test_y = np.load('./data/test_y.npy')
train_x = np.load('./data/train_x.npy')
train_y = np.load('./data/train_y.npy')
val_x = np.load('./data/val_x.npy')
val_y = np.load('./data/val_y.npy')


with open('./data/reviews.txt', 'r') as fd:
          reviews = fd.read()

reviews = reviews.lower()
all_text = ''.join([i for i in reviews if i not in punctuation])

reviews_split = all_text.split('\n')
reviews_split.pop()
all_text = ' '.join(reviews_split)

words = all_text.split()

       #print(len(all_text))
        #print(all_text[:100])

counts = Counter(words)
vocab = sorted(counts, key = counts.get, reverse = True)
vocab2int = {word: i for i, word in enumerate(vocab, start = 1 ) }

reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab2int[word] for word in review.split() ])

def pad_features(reviews_ints, seq_length):
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features



# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# RNN Model

import torch.nn as nn

class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size,
                    self.hidden_dim).zero_(),
                    weight.new(self.n_layers, batch_size,
                                self.hidden_dim).zero_()   )

        return hidden


# Inst  model
vocab2int_length = 74072    # len( vocab2int )
vocab_size = vocab2int_length + 1  # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

#training parameter
epochs = 4
counter = 0
print_every = 100   # for training status
clip=5              # gradient clipping

net.load_state_dict(torch.load('net_dict.pth'))
#pdb.set_trace()
net.eval()


# Testing data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in tqdm(test_loader):

    h = tuple([each.data for each in h])


    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss

print("Test data predictions !!!!!")
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])

    test_words = test_text.split()

    test_ints = []
    test_ints.append([vocab2int[word] for word in test_words])

    return test_ints


test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back.'

test_ints = tokenize_review(test_review_neg)
print(test_ints)

seq_length=200
features = pad_features(test_ints, seq_length)

print(features)

feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())


def predict(net, test_review, sequence_length=200):     #input model, input, seq_len
    net.eval()

    test_ints = tokenize_review(test_review)

    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)

    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    h = net.init_hidden(batch_size)

    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding

    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")









