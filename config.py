NUM_EPOCHS = 100
HIDDEN_SIZE = 256
NUM_LAYERS = 2
ENCODER_DROPOUT = 0.5
DECODER_DROPOUT = 0.5
BATCH_SIZE = 128
LR = 0.0001
TEACHER_FORCING_EPOCHS = 20
DEVICE = "cpu"

TRAIN_PATH = "data/train_hinglish.txt"
TEST_PATH = "data/test_hinglish.txt"
VOCAB_PATH_HINDI = "data/vocab_hindi.pickle"
VOCAB_PATH_ENGLISH = "data/vocab_english.pickle"
