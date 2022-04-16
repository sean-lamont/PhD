from autoencoder.autoencoder import *

# checkpoint_path = "autoencoder/models/2020_04_22_20_36_50" # os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)

# # logging.info("loading checkpoint from {}".format(checkpoint_path))

# checkpoint = Checkpoint.load(checkpoint_path)
# seq2seq = checkpoint.model
# input_vocab = checkpoint.input_vocab
# output_vocab = checkpoint.output_vocab

# batch_encoder = BatchPredictor(seq2seq, input_vocab, output_vocab)

# predictor = BatchPredictor(seq2seq, input_vocab, output_vocab)


seqs = ["@ @ Cmin$= CConseqConv$ASM_MARKER | Vy | Vx Vx", "@ @ Cmin$= CConseqConv$ASM_MARKER | Vy | Vy Vt", "@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2","@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2","@ Cbool$! | Vh1 @ Cbool$! | Vh2 @ @ Cmin$==> @ Cbool$~ @ @ Cmin$= Vh1 Vh2 @ Cbool$! | Vl1 @ Cbool$! | Vl2 @ Cbool$~ @ @ Cmin$= @ @ Clist$CONS Vh1 Vl1 @ @ Clist$CONS Vh2 Vl2"]

seqs = [i.strip().split() for i in seqs]

# relatively fast encoder: 2~5ms a batch
# compared to HOL's 0.1s time limit of tactic execution, encoding is not a bottleneck now

# print(predictor.predict(seqs))

s1 = timeit.default_timer()

# size: (2 [by default bidirectional], batch_num, hidden_length)
out, sizes = batch_encoder.encode(seqs)
representation = torch.cat(out.split(1), dim=2).squeeze()
print(representation)
print(representation.shape)
# print(out)

s2 = timeit.default_timer()
print(s2-s1)

