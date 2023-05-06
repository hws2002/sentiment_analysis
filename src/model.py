#%% import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# paramaters for model
class model_config():
    """
    For all datasets we use: rectified linear units, 
    filter windows (h) of 3, 4, 5 with 100 feature maps each, 
    dropout rate (p) of 0.5, l2 constraint (s) of 3, 
    and mini-batch size of 50. 
    These values were chosen via a grid search on the SST-2 dev set.
    ...
    From the paper: https://arxiv.org/pdf/1408.5882.pdf
    """
    from utils import vocab,s_vectors
    update_w2v = True           # whether to update w2v 
    vocab_size = len(vocab)+1   # +1 for padding (recall that we added one more row for sentence vector)
    n_classes = 2               # 0 -> neg, 1 -> pos | binary classification
    embedding_dim = 50          # dimension of word embedding. same as word2vec model length 50
    dropout_rate = 0.5          # dropout rate
    kernel_num = 20             # number of each kind of kernel
    kernel_sizes = [3,4,5]      # size of kernel, h (window size)
    pretrained_embed = s_vectors# pretrained embedding matrix
    #------------- RNN ONLY -----------------------------------------------------------------
    hidden_size = 100           # hidden size of rnn
    num_layers = 2              # number of layers of rnn

config = model_config()

#%% CNN model
class CNN(nn.Module):
    def __init__(self, config : model_config):
        super(CNN,self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_classes
        embedding_dim = config.embedding_dim
        dropout_rate = config.dropout_rate
        kernel_num = config.kernel_num
        kernel_sizes = config.kernel_sizes
        pretrained_embed = config.pretrained_embed

        # embedding layer
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        
        # convolution layer
        # input channel size is 1, because we only have one channel (word embedding) 
        # kernel_size = height * width!
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels= kernel_num,kernel_size=(kernel_sizes[0],embedding_dim),stride=1,padding = 0)
        self.conv1_2 = nn.Conv2d(1,kernel_num,(kernel_sizes[1],embedding_dim))
        self.conv1_3 = nn.Conv2d(1,kernel_num,(kernel_sizes[2],embedding_dim))
        
        # pooling layer
        self.pool = nn.MaxPool1d
        
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        # fully connected layer
        self.fc = nn.Linear(len(kernel_sizes)*kernel_num,n_class)

    @staticmethod
    def conv_and_pool(x,conv):
        x = conv(x)
        x = F.relu(x.squeeze(3)) #  concatenates 20
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self,x):
        # (batch_size,1,max_length,embedding_dim), converts sentence represented by id into batch size tensor
        x = self.embedding(x.to(torch.int64)).unsqueeze(1)
        x1 = self.conv_and_pool(x,self.conv1_1) # (batch_size, kernel_num)
        x2 = self.conv_and_pool(x,self.conv1_2) # (batch_size, kernel_num)
        x3 = self.conv_and_pool(x,self.conv1_3) # (batch_size, kernel_num)
        # concatenate x1,x2,x3 column-wise, apply dropout, and apply fully-connected layer to get output
        # as it's a binary classification, we use log_softmax as activation function
        x = F.log_softmax(self.fc(self.dropout(torch.cat((x1,x2,x3),1))),dim=1)
        return x
    
#%% RNN model
class RNN_LSTM(nn.Module):
    def __init__(self, config):

        super(RNN_LSTM, self).__init__()

        vocab_size = config.vocab_size
        update_w2v = config.update_w2v
        embedding_dim = config.embedding_dim
        pretrained_embed = config.pretrained_embed
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.__name__ = 'RNN_LSTM'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #! embedding is a table, which is used to lookup the embedding vector of a word
        self.embedding.weight.requires_grad = update_w2v
        #! if update_w2v is True, the embedding.weight will be updated during training
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        #! import the pretrained embedding vector as embedding.weight

        # (seq_len, batch, embed_dim) -> (seq_len, batch, 2 * hidden_size)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        # (batch, hidden_size * 2) -> (batch, num_classes)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        self.fc1 = nn.Linear(64, self.n_class)
        # (batch, num_classes) -> (batch, num_classes)

    def forward(self, inputs):
        _, (h_n, _) = self.encoder(self.embedding(inputs.to(torch.int64)).permute(1, 0, 2))  # (num_layers * 2, batch, hidden_size)
        # view h_n as (num_layers, num_directions, batch, hidden_size)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        return (self.fc1(self.decoder(torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1))))


#%% MLP model
