import pandas as pd
import matplotlib.pyplot as plt


def import_dataframe(path, shuffle=True):
    frame = pd.read_csv(path)
    if shuffle:
        frame.sample(frac=1).reset_index(drop=True)
    frame['GalaxyID'] = frame['GalaxyID'].astype(str) + '.jpg'
    return frame


def plot_history(history, plot_path):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig(plot_path)
