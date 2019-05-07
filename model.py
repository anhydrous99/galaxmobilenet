from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, AveragePooling2D, \
    Dense, ZeroPadding2D, Flatten, Add

# -------------------- Mobile Net V1 -------------------


def convolution_block(input_tensor, filters, kernel_size=(1, 1), strides=(1, 1), padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def depth_wise_convolution_block(input_tensor, filters, depth_wise_strides, depthwise_padding):
    x = DepthwiseConv2D((3, 3), depth_wise_strides, padding=depthwise_padding, use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return convolution_block(x, filters)


def mobile_net(input_tensor, n_classes):
    x = ZeroPadding2D(((0, 1), (0, 1)))(input_tensor)
    x = convolution_block(x, 32, (3, 3), (2, 2), padding='valid')
    x = depth_wise_convolution_block(x, 64, (1, 1), 'same')
    x = ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = depth_wise_convolution_block(x, 128, (2, 2), 'valid')
    x = depth_wise_convolution_block(x, 128, (1, 1), 'same')
    x = ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = depth_wise_convolution_block(x, 256, (2, 2), 'valid')
    x = depth_wise_convolution_block(x, 256, (1, 1), 'same')
    x = ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = depth_wise_convolution_block(x, 512, (2, 2), 'valid')
    for i in range(5):
        x = depth_wise_convolution_block(x, 512, (1, 1), 'same')
    x = ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = depth_wise_convolution_block(x, 1024, (2, 2), 'valid')
    x = ZeroPadding2D(((3, 3), (3, 3)))(x)
    x = depth_wise_convolution_block(x, 1024, (2, 2), 'same')
    x = AveragePooling2D((7, 7), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(1000)(x)
    x = Activation('relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    return x

# -------------------- Mobile Net V2 -------------------


def bottleneck(input_tensor, filters, depth_wise_strides, expansion_factor=6):
    expansion = input_tensor.shape[3].value * expansion_factor
    x = convolution_block(input_tensor, expansion)
    x = depth_wise_convolution_block(x, expansion, depth_wise_strides, 'same')
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    if depth_wise_strides == (1, 1) and filters == input_tensor.shape[3].value:
        x = Add()([x, input_tensor])
    return x


def mobile_net_v2(input_tensor, n_classes):
    x = ZeroPadding2D(((0, 1), (0, 1)))(input_tensor)
    x = convolution_block(x, 32, (3, 3), (2, 2), 'valid')
    x = bottleneck(x, 16, (1, 1), 1)
    x = bottleneck(x, 24, (2, 2))
    x = bottleneck(x, 24, (1, 1))
    x = bottleneck(x, 32, (2, 2))
    x = bottleneck(x, 32, (1, 1))
    x = bottleneck(x, 32, (1, 1))
    x = bottleneck(x, 64, (2, 2))
    for i in range(3):
        x = bottleneck(x, 64, (1, 1))
    for i in range(3):
        x = bottleneck(x, 96, (1, 1))
    x = bottleneck(x, 160, (2, 2))
    x = bottleneck(x, 160, (1, 1))
    x = bottleneck(x, 160, (1, 1))
    x = bottleneck(x, 320, (1, 1))
    x = convolution_block(x, 1280, (1, 1), (1, 1), 'same')
    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dense(1000)(x)
    x = Activation('relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    return x
