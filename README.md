# Hinglish Transliterator

An LSTM-based encoder-decoder model for transliterating text from Romanized Hindi (Hinglish) to Devanagari script.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [To-Do](#to-do)
- [Contributing](#contributing)

## Introduction

This project aims to build a neural transliteration system that converts Hinglish text (Hindi written with English alphabets) into Hindi written in Devanagari script. We use a sequence-to-sequence learning model with LSTM networks for this purpose.

## Features

- Transliteration of Hinglish text to Hindi script.
- LSTM-based encoder-decoder architecture.
- Support for variable-length input and output sequences.
- Beam search decoding for improved transliteration accuracy.

## Installation

To install and run this project, follow these steps:

```
git clone https://github.com/adityasihag1996/Hinglish.git
cd hinglish-transliterator
pip install -r requirements.txt
```

## Usage

**_NOTE:-_** Before running the scripts, please adjust the paths accordingly in `config.py`.
**_A sample checkpoint has been provided in the repo for testing, `seq2seq_hinglish.pth`_**

To transliterate text, run the following command:

```
python transliterate.py --input "your input text here" -mp "/path/to/your/model" -ev "/path/to/your/english_vocab.pickle" -mp "/path/to/your/hindi_vocab.pickle"
```
The script transliterate.py will output the transliterated text in Devanagari script.

For training the model:

```
python train.py
```
This will start the training process using the default hyperparameters defined in the script. Checkpoints will be saved in the checkpoints directory.

**_DATASET_:-** Dataset has been curated using `Google Dakshina dataset <https://github.com/google-research-datasets/dakshina>`__
You will find partial train and test files used to train this model, in the `/data` directory.

## Sample Inference
Below is a sample sequence, and model prediction:-

`अखंडानंद` -> `akhandanand`

## To-Do

- [ ] Add cross attention between Encoder and Decoder.
- [ ] Transformer based models.

## Contributing
Contributions to improve the project are welcome. Please follow these steps to contribute:

Fork the repository.
Create a new branch for each feature or improvement.
Submit a pull request with a comprehensive description of changes.