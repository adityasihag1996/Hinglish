def read_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    hindi_words, english_words = [], []
    for line in lines:
        hindi_word, english_word = line.split(',')  # Assuming the delimiter is a comma
        hindi_words.append(hindi_word)
        english_words.append(english_word)
    return hindi_words, english_words

