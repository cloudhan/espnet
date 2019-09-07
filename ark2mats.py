import os
import argparse
import kaldiio

parser = argparse.ArgumentParser()
parser.add_argument("ark_file")
parser.add_argument("out_dir")
args = parser.parse_args()
print(args)

ark = kaldiio.load_ark(args.ark_file)

os.makedirs(args.out_dir, exist_ok=True)

for name, feat in ark:
    ofile = os.path.realpath(os.path.join(args.out_dir, name))
    kaldiio.save_mat(ofile, feat)
