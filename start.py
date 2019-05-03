import argparse
import os

parser = argparse.ArgumentParser(
    description='Creates and trains a small Convolutional Neural Network to classify images of flowers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of times to train over data.'
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='How many images to train on at a time.'
)
parser.add_argument(
    '--percentage_validation',
    type=int,
    default=10,
    help='Percentage of data to use as validation'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='Rate at which to train the model'
)
parser.add_argument(
    '--save_hdf5',
    type=str,
    default='',
    help='Saves the model as an HDF5 file'
)
parser.add_argument(
    '--save_history_plot',
    type=str,
    default='',
    help='Save a plot of the model training'
)

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
percentage_validation = args.percentage_validation
learning_Rate = args.learning_rate
hdf5_path = args.save_hdf5
plot_path = args.save_history_plot
