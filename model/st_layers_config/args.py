import numpy as np
import pandas as pd
import configparser
import argparse
import os

curPath = os.path.abspath(os.path.dirname(__file__))
print('ST_Encoder', curPath)

def parse_args():
    # get configuration
    config_file = curPath + '/ST_Encoder.conf'
    config = configparser.ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')

    # model
    parser.add_argument('--dilation_exponential', type=int, default=config['model']['dilation_exponential'])
    parser.add_argument('--conv_channels', type=int, default=config['model']['conv_channels'])
    parser.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
    parser.add_argument('--skip_channels', type=int, default=config['model']['skip_channels'])
    parser.add_argument('--end_channels', type=int, default=config['model']['end_channels'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])

    args, _ = parser.parse_known_args()

    args.adj_mx = None
    return args