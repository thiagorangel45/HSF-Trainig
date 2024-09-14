from get_dataset import get_dataset
import argparse


parser = argparse.ArgumentParser(description = '')
parser.add_argument('-d', '--dataset',
                    required = True, type=str,
                    help='Name of the toy dataset')

parser.add_argument('-n', '--npoints',
                    required = False, default = 200000, type=int,
                    help='Number of points')

args = vars(parser.parse_args())
input_dataset = args['dataset']
npoints = args['npoints']

get_dataset(input_dataset, npoints)
