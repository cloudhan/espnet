import os
import sys
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("feat", help="dumped feature from C++")
parser.add_argument("feat_dim", type=int)
parser.add_argument("num_words", type=int)
parser.add_argument("am", help="am model, something like model.acc.best")
parser.add_argument("lm", help="lm model, something like model.acc.best")
parser.add_argument("words", help="dict, something like train_sp_units.txt")

parser.add_argument("--ctc-weight",  default=0.5)
parser.add_argument("--lm-weight", default=0.7)

parser.add_argument("--output-dir", "-o", default=".", help="dir for dumping data")
parser.add_argument("--dump-encoder", "-de", default=False, action="store_true", help="dump encoder result")
parser.add_argument("--dump-decoder", "-dd", default=False, action="store_true", help="dump decoder result")
parser.add_argument("--dump-ctc-prob", "-dcp", default=False, action="store_true", help="dump ctc prob")
args = parser.parse_args()

import torch
import numpy as np

import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.lm.default import DefaultRNNLM
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.pytorch_backend.ctc import CTC





root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)


def dump(array, f):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f)
    if isinstance(array, torch.Tensor):
        array.detach().numpy().tofile(path)
    elif isinstance(array, np.ndarray):
        array.tofile(path)
    else:
        raise RuntimeError("unknown")
    print("Dumped", path)


################################ init feature ################################
data = np.fromfile(args.feat, "<f4").reshape((-1, args.feat_dim))
data = torch.from_numpy(data)
data = data.unsqueeze(0)

################################ load Encoder and Decoder params ################################
model = torch.load(args.am, map_location='cpu')
model.eos = args.num_words - 1
model.sos = model.eos

edict = {}
ddict = {}
for k, v in model.items():
    if "encoder" in k:
        key = k.replace("encoder.", "")
        # print(key,v.shape)
        edict[key] = v

    if "decoder" in k:
        key = k.replace("decoder.", "")
        # print(key,v.shape)
        ddict[key] = v


################################ init Encoder ################################
encoder = Encoder(idim=args.feat_dim, attention_dim=256, attention_heads=4,
                  linear_units=2048, num_blocks=12, input_layer="conv2d",
                  dropout_rate=0.5, positional_dropout_rate=0.5,
                  attention_dropout_rate=0.5)
encoder.eval()
encoder.load_state_dict(edict)

################################ init decoder ################################
decoder = Decoder(odim=args.num_words, attention_dim=256, attention_heads=4,
                  linear_units=2048, num_blocks=6, dropout_rate=0.5,
                  positional_dropout_rate=0.5, self_attention_dropout_rate=0.5,
                  src_attention_dropout_rate=0.5)

decoder.eval()
decoder.load_state_dict(ddict)


enc, _ = encoder.forward(data, None)

print(enc.shape)
assert enc.shape[0] == 1  # FIXME:

if args.dump_encoder:
    dump(enc, "encoder.bin")

if args.dump_ctc_prob:
    ctc = F.linear(enc, model["ctc.ctc_lo.weight"], model["ctc.ctc_lo.bias"])
    softmax = F.softmax(ctc, -1)
    dump(softmax, "ctc_prob.bin")


x = torch.tensor([[args.num_words - 1, 1665]])
dec = decoder.forward_one_step(x, None, enc)
# print(len(dec))
# topk = torch.topk(dec, 16)
# print("decoder result:", dec, "shape: ", dec.shape)
# print("topk:", topk)

if args.dump_decoder:
    dump(dec[0], "decoder0.bin")
    dump(dec[1], "decoder1.bin")


################################ init lm ################################
lm_model = torch.load(args.lm, map_location='cpu')
lmdict = {}
for k, v in lm_model.items():
    if(k.endswith("weight") or k.endswith("bias")):
        lmdict[k] = v
    else:
        # reorder, predictor.rnn.weight_ih_l2 -> predictor.rnn.2.weight_ih
        prefix, suffix = k.rsplit("_", 1)
        suffix = suffix.strip("l")
        prefix, medium = prefix.rsplit(".", 1)
        key = ".".join([prefix, suffix, medium])
        lmdict[key] = v

lm_parser = argparse.ArgumentParser()
lm_args = DefaultRNNLM.add_arguments(lm_parser).parse_args([])
lm_args.layer = 3
lm_args.unit = 1100
lm_args.embed_unit = 250
lm = DefaultRNNLM(args.num_words, lm_args)
lm.eval()
lm.load_state_dict(lmdict)


def load_char_list(path):
    char_list = ["<blank>"]
    with open(path) as f:
        for l in f.readlines():
            word, idx = l.split(" ")
            # print(word, idx)
            char_list.append(word)
    char_list.append("<eos>")
    return char_list


char_list = load_char_list(args.words)

ctc = CTC(args.num_words, 256, 0.0, ctc_type="builtin")
ctc.load_state_dict({
    "ctc_lo.weight": model["ctc.ctc_lo.weight"],
    "ctc_lo.bias": model["ctc.ctc_lo.bias"]
})
ctc_scorer = CTCPrefixScorer(ctc, model.eos)

beam_search = BeamSearch(
    pre_beam_ratio=1.5,
    beam_size=10,
    vocab_size=args.num_words,
    weights={
        "ctc": args.ctc_weight,
        "decoder": 1.0 - args.ctc_weight,
        "lm": args.lm_weight
    },
    scorers={
        "ctc": ctc_scorer,
        "decoder": decoder,
        "lm": lm
    },
    sos=model.sos,
    eos=model.eos,
    token_list=char_list,
)
beam_search.eval()

# assert (ctc.shape[0] == 1)
# ctc = ctc.reshape((399, 256))

assert (enc.shape[0] == 1)
enc = enc.reshape((-1, 256))

nbest = beam_search.forward(x=enc, maxlenratio=0.0)
nbest.sort(key=lambda x: -x.score)
for h in nbest:
    print(h.score, list(map(lambda x: char_list[x], h.yseq)))
