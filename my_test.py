
import torch
import numpy as np
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.lm.default import DefaultRNNLM
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.pytorch_backend.ctc import CTC

import sys
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)


num_words = 7244

model = torch.load(
    "/home/hanguangyun/models/asr_model_1113/am_model/20191113_htrs760h+goog1000h+yitu200h_ep4/model.acc.best", map_location='cpu')

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


encoder = Encoder(idim=83, attention_dim=256, attention_heads=4,
                  linear_units=2048, num_blocks=12, input_layer="conv2d",
                  dropout_rate=0.5, positional_dropout_rate=0.5,
                  attention_dropout_rate=0.5)

encoder.eval()
encoder.load_state_dict(edict)

import kaldiio
# data = kaldiio.load_mat("/data1/hanguangyun/workspaces/dump/test/mats/KmoUAF0b-9SEf8tfAAAAANv7mq4268.raw")

data = np.fromfile("/home/hanguangyun/fbank.bin", "<f4").reshape((-1, 83))

data = torch.from_numpy(data)
data = data.unsqueeze(0)

enc, _ = encoder.forward(data, None)
# print(enc)
enc.detach().numpy().tofile("encoder.bin")
# ctc = F.linear(enc,model["ctc.ctc_lo.weight"],model["ctc.ctc_lo.bias"])
# softmax = F.softmax(ctc,-1)
# softmax.detach().numpy().tofile("ctc_prob.bin")
# print(softmax,softmax.shape)

decoder = Decoder(odim=num_words, attention_dim=256, attention_heads=4,
                  linear_units=2048, num_blocks=6, dropout_rate=0.5,
                  positional_dropout_rate=0.5, self_attention_dropout_rate=0.5,
                  src_attention_dropout_rate=0.5)

decoder.eval()
decoder.load_state_dict(ddict)

x = torch.tensor([[num_words - 1, 1665]])
dec = decoder.recognize(x, None, enc)
topk = torch.topk(dec, 16)
# print("decoder result:", dec, "shape: ", dec.shape)
# print("topk:", topk)
# dec.detach().numpy().tofile("dec.bin")

lm_model = torch.load(
    "/home/hanguangyun/models/asr_model_1113/lm_model/model.2.2g+politic.3gram-4gram.config_v3/rnnlm.model.best", map_location='cpu')
lmdict = {}
for k, v in lm_model.items():
    if(k.endswith("weight") or k.endswith("bias")):
        lmdict[k] = v
    else:
        prefix, suffix = k.rsplit("_", 1)
        suffix = suffix.strip("l")
        prefix, medium = prefix.rsplit(".", 1)
        key = ".".join([prefix, suffix, medium]) # reorder, predictor.rnn.weight_ih_l2 -> predictor.rnn.2.weight_ih
        lmdict[key] = v

import argparse
parser = argparse.ArgumentParser()
args = DefaultRNNLM.add_arguments(parser).parse_args()
args.layer = 3
args.unit = 250
lm = DefaultRNNLM(num_words, args)
lm.eval()
# print(lmdict.keys())
# print(lm.state_dict().keys())
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


char_list = load_char_list("/data1/hanguangyun/models/asr_model_1113/am_model/20191113_htrs760h+goog1000h+yitu200h_ep4/train_sp_units.txt")
model.eos = num_words - 1
model.sos = model.eos

ctc = CTC(num_words, 256, 0.0, ctc_type="builtin")
ctc.load_state_dict({
    "ctc_lo.weight": model["ctc.ctc_lo.weight"],
    "ctc_lo.bias": model["ctc.ctc_lo.bias"]
})
ctc_scorer = CTCPrefixScorer(ctc, model.eos)

beam_search = BeamSearch(
    pre_beam_ratio=1.5,
    beam_size=10,
    vocab_size=num_words,
    weights={
        "ctc": 0.5,
        "decoder": 0.5,
        "lm": 0.3
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
