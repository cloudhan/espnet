#!/bin/bash

set -ev

/data/hanguangyun/workspaces/kaldi-cmake/build/src/featbin/compute-fbank-feats \
  --verbose=30000 --print-args=true \
  --config=/data/hanguangyun/workspaces/espnet/fbank.conf  \
  scp:/data/hanguangyun/workspaces/test.scp \
  ark:fbank.ark

/data/hanguangyun/workspaces/kaldi-cmake/build/src/featbin/compute-kaldi-pitch-feats \
  --sample-frequency=16000 \
  scp:/data/hanguangyun/workspaces/test.scp \
  ark:raw_pitch.ark

/data/hanguangyun/workspaces/kaldi-cmake/build/src/featbin/process-kaldi-pitch-feats \
  ark:raw_pitch.ark ark:pitch.ark

/data/hanguangyun/workspaces/kaldi-cmake/build/src/featbin/paste-feats \
  ark:fbank.ark ark:pitch.ark ark:test.ark

/data/hanguangyun/workspaces/kaldi-cmake/build/src/featbin/apply-cmvn \
  --norm-vars=true \
  /data/hanguangyun/workspaces/0918model/cmvn.ark \
  ark:test.ark ark:test2.ark
