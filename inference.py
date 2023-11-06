import torch
import argparse
import pickle

from model import Encoder, Decoder, Seq2Seq


def transliterate(input_text, encoder, decoder, source_vocab, target_vocab, target_vocab_inv, max_length = 50, device = "cpu"):
    # Turn the input text to lowercase
    input_text = input_text.lower()
    # Tokenize the input text into characters
    input_tokens = [source_vocab[char] for char in input_text]
    # Add the batch dimension
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    
    # Move to the same device as the model
    input_tensor = input_tensor.to(device)
    
    # Get the input length for pack_padded_sequence
    input_length = torch.tensor([len(input_tokens)]).to(device)
    
    # Encoder part
    with torch.no_grad():
        encoder_hidden = encoder.initHidden().to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_length, encoder_hidden)
    
    # Decoder part
    decoder_input = torch.tensor([[target_vocab['<SOS>']]], device=device)  # SOS
    decoder_hidden = encoder_hidden  # Use last hidden state from the encoder to start the decoder
    
    # Store the output words and attention weights
    decoded_words = []
    
    for di in range(max_length):
        with torch.no_grad():
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)  # Get most likely word index
            if topi.item() == target_vocab['<EOS>']:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_vocab_inv[topi.item()])
            
            # Next input is current output
            decoder_input = topi.squeeze().detach()

    return ''.join(decoded_words)


def parse_opt():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset of images with text.")

    parser.add_argument("-mp", "--model_path", type=str, required=True,
                    help="Path to the model state-dict.")
    parser.add_argument("-inp", "--input", type=str, required=True,
                    help="Input to be transliterated.")
    parser.add_argument("-ev", "--vocab_english", type=str, required=True,
                    help="Path to the english vocab pickle file.")
    parser.add_argument("-hv", "--vocab_hindi", type=str, required=True,
                    help="Path to the hindi vocab pickle file.")

    return parser.parse_args()



if __name__ == '__main__':
    # args
    args = parse_opt()

    input_seq = args.input
    model_path = args.model_path
    hindi_vocab_path = args.vocab_hindi
    english_vocab_path = args.vocab_english

    with open(hindi_vocab_path, 'rb') as f:
        hindi_vocab = pickle.load(f)

    with open(english_vocab_path, 'rb') as f:
        english_vocab = pickle.load(f)

    english_vocab_inv = {idx: char for char, idx in english_vocab.items()}

    model = Seq2Seq(Encoder, Decoder, english_vocab['<SOS>'])
    model.load_state_dict(torch.load(model_path))

    transliterated_text = transliterate(input_seq, model.encoder, model.decoder, hindi_vocab, english_vocab, english_vocab_inv, device = "cpu")
    print(transliterated_text)