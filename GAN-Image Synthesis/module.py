import tensorflow_addons as tfa
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    # Comment by K.C: selection different method of Normalization, here, were are using batch_normalization
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tfa.layers.LayerNormalization

# Comment by K.C:
# ConvGenerator defines the attributes of a generator

def ConvGenerator(input_shape=(1, 1, 128),
                  output_channels=3,
                  dim=64,
                  n_upsamplings=4,
                  norm='batch_norm',
                  name='ConvGenerator'):
    Norm = _get_norm_layer(norm)

    # 0
    # Comment by K.C: h is the noise input with the length 128, it goes through a deconvolution pad and upgraded from 1
    # to 64*2^(n_upsampling-1), the minimize pad size is 64*8=512

    h = inputs = keras.Input(shape=input_shape)

    # 1: 1x1 -> 4x4
    # Comment by K.C: d = min(dim * 2^(n_upsampling - 1), dim * 8)
    d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
    h = keras.layers.Conv2DTranspose(d, 4, strides=1, padding='valid', use_bias=False)(h)
    # Comment by K.C: this is a deconvolution layer (Transpose convolution layer), upsamples the noise signal to at
    # most 512x512
    h = Norm()(h) # selection the Normalization tye based on the Norm function
    h = keras.layers.ReLU()(h)

    # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
        h = keras.layers.Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.ReLU()(h)

    h = keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', use_bias=False)(h)
    h = keras.layers.Activation('tanh')(h)
    # Added by K.C:
    # pringt the structure of the generator
    # convG = keras.Model(inputs=inputs, outputs=h, name=name)
    # keras.utils.plot_model(convG,'output/convGenerator.png', show_shapes=True)
    return keras.Model(inputs=inputs, outputs=h, name=name)


def ConvDiscriminator(input_shape=(64, 64, 3),
                      dim=64,
                      n_downsamplings=4,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = keras.layers.LeakyReLU(alpha=0.2)(h)

    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = keras.layers.Conv2D(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3: logit
    h = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(h)
    # Added by K.C:
    # Print the structure of the discriminator
    # convD = keras.Model(inputs=inputs, outputs=h, name=name)
    # keras.utils.plot_model(convD,'output/convDiscriminator.png', show_shapes=True)
    return keras.Model(inputs=inputs, outputs=h, name=name)
