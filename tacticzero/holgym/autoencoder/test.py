import os
import argparse
import logging
import timeit

# import torch
# import torchtext

# import seq2seq
# from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from predictor import Predictor
from checkpoint import Checkpoint


checkpoint_path = "models/2020_04_21_23_23_15" # os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
logging.info("loading checkpoint from {}".format(checkpoint_path))

checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

predictor = Predictor(seq2seq, input_vocab, output_vocab)

# l1 = ["1" for _ in range(100)]
# l2 = ["1" for _ in range(51)]

# print(predictor.encode(l1))
# print(predictor.encode(l2))    

while True:
    seq_str = input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
    start = timeit.default_timer()
    print(predictor.encode(seq))
    end = timeit.default_timer()
    print(end-start)
