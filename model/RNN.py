import torch.nn as nn
from model import Model
import torch
from torch.utils.data import DataLoader


class RNN(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(RNN).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.actv = nn.Sigmoid()

    def forward(self, text, text_len):
        # each text is of the format :
        # text = [batch size,sent_length]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(text, text_len, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs

    def p(self, data: torch.Tensor) -> torch.Tensor:
        # deactivating dropout layers
        self.eval()

        predictions: torch.Tensor

        # deactivates autograd
        with torch.no_grad():
            # retrieve text and no. of words
            text = data
            text_length = torch.zeros(size=(len(data), 1))
            for i in range(len(data)):
                text_length[i] = len(text[i])

            # convert to 1d tensor
            predictions = self(text, text_length).squeeze()

        self.train()

        return predictions
