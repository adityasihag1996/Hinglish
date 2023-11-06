import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input_seq, input_lengths):
        embedded = self.embedding(input_seq)
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_step, hidden):
        embedded = self.embedding(input_step)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, SOS_token):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.SOS_token = SOS_token
        
    def forward(self, source_seq, source_lens, target_seq, target_lens, teacher_forcing):
        batch_size = source_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        encoder_outputs, hidden = self.encoder(source_seq, source_lens)
        
        # First input to the decoder is the <SOS> tokens
        decoder_input = torch.tensor([self.SOS_token] * batch_size)
        
        for t in range(1, target_len):
            decoder_output, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)
            outputs[:, t] = decoder_output
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1)

            # Teacher forcing: next input is current target
            if teacher_forcing:
                decoder_input = target_seq[:, t]
        
        return outputs