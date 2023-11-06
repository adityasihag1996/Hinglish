import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from utils import read_data
from config import TRAIN_PATH, TEST_PATH, VOCAB_PATH_HINDI, VOCAB_PATH_ENGLISH, BATCH_SIZE


class TransliterationDataset(Dataset):
    def __init__(self, hindi_words, english_words, hindi_vocab, english_vocab):
        self.hindi_words = hindi_words
        self.english_words = english_words
        self.hindi_vocab = hindi_vocab
        self.english_vocab = english_vocab

    def __len__(self):
        return len(self.hindi_words)

    def __getitem__(self, idx):
        hindi_word = self.hindi_words[idx]
        english_word = self.english_words[idx]
        hindi_indices = [self.hindi_vocab['<SOS>']] + [self.hindi_vocab[char] for char in hindi_word] + [self.hindi_vocab['<EOS>']]
        english_indices = [self.english_vocab[char] for char in english_word] + [self.english_vocab['<EOS>']]
        
        return (
            torch.tensor(hindi_indices, dtype=torch.long), 
            torch.tensor(english_indices, dtype=torch.long), 
            len(hindi_indices),  # changed to length of hindi_indices
            len(english_indices)  # changed to length of english_indices
        )

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True) # sort by length of hindi word for efficient packing
    hindi_seqs, english_seqs, hindi_lens, english_lens = zip(*batch)

    # Padding sequences with '<PAD>' index
    hindi_seqs_padded = torch.nn.utils.rnn.pad_sequence(hindi_seqs, batch_first=True, padding_value = HINDI_PAD_TOKEN)
    english_seqs_padded = torch.nn.utils.rnn.pad_sequence(english_seqs, batch_first=True, padding_value = ENGLISH_PAD_TOKEN)

    return hindi_seqs_padded, torch.tensor(hindi_lens), english_seqs_padded, torch.tensor(english_lens)


ENGLISH_PAD_TOKEN = None
HINDI_PAD_TOKEN = None

def create_dataset_and_dataloader():
    global ENGLISH_PAD_TOKEN, HINDI_PAD_TOKEN

    with open(VOCAB_PATH_HINDI, 'rb') as f:
        hindi_vocab = pickle.load(f)

    with open(VOCAB_PATH_ENGLISH, 'rb') as f:
        english_vocab = pickle.load(f)

    train_hindi_words, train_english_words = read_data(TRAIN_PATH)
    test_hindi_words, test_english_words = read_data(TEST_PATH)

    ENGLISH_PAD_TOKEN = english_vocab['<PAD>']
    HINDI_PAD_TOKEN = hindi_vocab['<PAD>']

    # Assuming 'train_hindi_words' and 'train_english_words' are already defined lists of words
    train_dataset = TransliterationDataset(train_hindi_words, train_english_words, hindi_vocab, english_vocab)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # Assuming 'test_hindi_words' and 'test_english_words' are already defined lists of words
    test_dataset = TransliterationDataset(test_hindi_words, test_english_words, hindi_vocab, english_vocab)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    return train_loader, test_loader, english_vocab, hindi_vocab
