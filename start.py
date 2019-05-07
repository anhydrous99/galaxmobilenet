import argparse
from create_train import create_and_train
from utils import import_dataframe, plot_history

parser = argparse.ArgumentParser(
    description='Creates and trains a small Convolutional Neural Network to classify images of flowers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    help='Number of times to train over data.'
)
parser.add_argument(
    '--text_path',
    type=str,
    default='Data/training_solutions_rev1.csv',
    help='Path to data folder'
)
parser.add_argument(
    '--image_dir',
    type=str,
    default='Data/images_training_rev1'
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
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
    default=0.0005,
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
text_path = args.text_path
image_dir = args.image_dir
batch_size = args.batch_size
percentage_validation = args.percentage_validation
learning_Rate = args.learning_rate
hdf5_path = args.save_hdf5
plot_path = args.save_history_plot

data_frame = import_dataframe(text_path)

model, history = create_and_train(data_frame, image_dir, batch_size, epochs, percentage_validation,
                                  learning_Rate, 36)

if hdf5_path:
    model.save(hdf5_path)

if plot_path:
    plot_history(history, plot_path)
