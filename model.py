from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, AveragePooling2D, Dense


def convolution_block(input_tensor, filters, kernel_size=(1, 1), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides)(input_tensor)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def depth_wise_convolution_block(input_tensor, filters, depth_wise_strides):
    x = DepthwiseConv2D((3, 3), depth_wise_strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return convolution_block(x, filters)


def mobile_net(input_tensor, n_classes):
    x = convolution_block(input_tensor, 32, (3, 3), (2, 2))
    x = depth_wise_convolution_block(x, 64, (1, 1))
    x = depth_wise_convolution_block(x, 128, (2, 2))
    x = depth_wise_convolution_block(x, 128, (1, 1))
    x = depth_wise_convolution_block(x, 256, (2, 2))
    x = depth_wise_convolution_block(x, 256, (1, 1))
    x = depth_wise_convolution_block(x, 512, (2, 2))
    for i in range(5):
        x = depth_wise_convolution_block(x, 512, (1, 1))
    x = depth_wise_convolution_block(x, 1024, (2, 2))
    x = depth_wise_convolution_block(x, 1024, (2, 2))
    x = AveragePooling2D((7, 7))(x)
    x = Dense(1000)(x)
    x = Activation('relu')(x)
    x = Dense(n_classes, 'softmax')(x)
    return x
