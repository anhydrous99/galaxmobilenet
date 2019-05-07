from model import mobile_net
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam


def create_and_train(data_frame, image_dir, batch_size, epochs, perc_valid, lr, n_classes):
    # Create model
    init = Input(shape=(224, 224, 3))
    x = mobile_net(init, n_classes)
    model = Model(inputs=init, outputs=x, name="GalaxyMobileNetv1")

    opt = Adam(lr=lr)

    classes = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1',
               'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.3', 'Class5.4',
               'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1',
               'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
               'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3',
               'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6']

    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)

    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['accuracy']
    )

    generator = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        shear_range=0.3,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 255.,
        validation_split=1./perc_valid
    )
    train_flow = generator.flow_from_dataframe(
        dataframe=data_frame,
        x_col='GalaxyID',
        y_col=classes,
        class_mode='other',
        target_size=(224, 224),
        batch_size=batch_size,
        directory=image_dir,
        subset='training'
    )
    valid_flow = generator.flow_from_dataframe(
        dataframe=data_frame,
        x_col='GalaxyID',
        y_col=classes,
        class_mode='other',
        target_size=(224, 224),
        batch_size=batch_size,
        directory=image_dir,
        subset='validation'
    )
    history = model.fit_generator(
        generator=train_flow,
        steps_per_epoch=train_flow.n // train_flow.batch_size,
        validation_data=valid_flow,
        validation_steps=3 * (valid_flow.n // valid_flow.batch_size),
        callbacks=[reduce_lr],
        epochs=epochs
    )
    return model, history
