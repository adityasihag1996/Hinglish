import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss

from model import Encoder, Decoder, Seq2Seq
from dataset import create_dataset_and_dataloader
from config import NUM_EPOCHS, LR, DEVICE, NUM_LAYERS, HIDDEN_SIZE, NUM_EPOCHS, TEACHER_FORCING_EPOCHS


def runner(model, train_loader, num_epochs, learning_rate, device, english_pad_token):
    criterion = nn.CrossEntropyLoss(ignore_index = english_pad_token)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        if epoch < TEACHER_FORCING_EPOCHS:
            teacher_forcing = True
        else:
            teacher_forcing = False

        for batch in train_loader:
            source_seqs, source_lens, target_seqs, target_lens = batch
            source_seqs, source_lens = source_seqs.to(device), source_lens.to(device)
            target_seqs, target_lens = target_seqs.to(device), target_lens.to(device)

            optimizer.zero_grad()

            # Forward pass through encoder-decoder
            # The decoder's output is now a sequence of predictions for each time step
            output = model(source_seqs, source_lens, target_seqs, target_lens, teacher_forcing)

            # Calculate loss
            # Ignore the first token of the target sequence (which is <SOS>) in loss calculation
            loss = criterion(output[:, 1:].reshape(-1, OUTPUT_SIZE), target_seqs[:, 1:].reshape(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

        torch.save(model.state_dict(), f'seq2seq_hinglish_e{epoch}.pth')

    print("Training Complete!")



if __name__ == '__main__':
    train_loader, test_loader, english_vocab, hindi_vocab = create_dataset_and_dataloader()

    INPUT_SIZE = len(hindi_vocab)+1
    OUTPUT_SIZE = len(english_vocab)+1

    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    decoder = Decoder(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, english_vocab["<SOS>"])

    runner(model, train_loader, NUM_EPOCHS, LR, DEVICE, english_vocab["<PAD>"])

