import os
import argparse
import logging
import timeit
import torch

# from seq2seq.evaluator import BatchPredictor
# from seq2seq.util.checkpoint import Checkpoint

from autoencoder.batch_predictor import BatchPredictor
from autoencoder.checkpoint import Checkpoint

# Load models directly because the package is installed
# checkpoint_path = "autoencoder/models/2020_04_22_20_36_50" # 91% accuracy model, only core theories
# checkpoint_path = "autoencoder/models/2020_04_26_20_11_28" # 95% accuracy model, core theories + integer + sorting
# checkpoint_path = "autoencoder/models/2020_09_24_23_38_06" # 98% accuracy model, core theories + integer + sorting | separate theory tokens
# checkpoint_path = "autoencoder/models/2020_11_28_16_45_10" # 96-98% accuracy model, core theories + integer + sorting + real | separate theory tokens

checkpoint_path = "autoencoder/models/2020_12_04_03_47_22" # 97% accuracy model, core theories + integer + sorting + real + bag | separate theory tokens

################## checkpoint_path = "autoencoder/models/2021_02_21_15_46_04" # 98% accuracy model, up to probability theory

# checkpoint_path = "autoencoder/models/2021_02_22_16_07_03" # 97-98% accuracy model, up to and include probability theory

# logging.info("loading checkpoint from {}".format(checkpoint_path))

checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

batch_encoder = BatchPredictor(seq2seq, input_vocab, output_vocab)

# predictor = BatchPredictor(seq2seq, input_vocab, output_vocab)

# seqs = ["@ @ Cmin$= CConseqConv$ASM_MARKER | Vy | Vx Vx", "@ @ Cmin$= CConseqConv$ASM_MARKER | Vy | Vy Vt", "@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2","@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2","@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2"]

# seqs = ["@ Cbool$~ @ @ @ Clist$SHORTLEX VR Vl Clist$NIL"]

# seqs = [i.strip().split() for i in seqs]

# # relatively fast encoder: 2~5ms a batch
# # compared to HOL's 0.1s time limit of tactic execution, encoding is not a bottleneck now

# print(batch_encoder.predict(seqs))

# s1 = timeit.default_timer()

# # size: (2 [by default bidirectional], batch_num, hidden_length)
# out, sizes = batch_encoder.encode(seqs)
# representation = torch.cat(out.split(1), dim=2).squeeze()
# print(representation)
# print(representation.shape)
# # print(out)

# s2 = timeit.default_timer()
# print(s2-s1)


# s1 = timeit.default_timer()
# # out, sizes = batch_encoder.encode(["@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2"])

# out, sizes = batch_encoder.encode(["@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2".strip().split()])

# # merge two hidden variables
# representations = torch.cat(out.split(1), dim=2).squeeze(0)
# print(representations)
# # print(representations.shape)
# s2 = timeit.default_timer()    
# print(s2-s1)
