import os
import argparse


parser = argparse.ArgumentParser(description='NI: Stas Action Classification') 

parser.add_argument('--gpu', type=int, required=True)

args = parser.parse_args()

os.system("export CUDA_VISIBLE_DEVICES=" + str(args.gpu))
