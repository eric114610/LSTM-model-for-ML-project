
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow as tf


def EEGNet(nb_classes, Chans = 64, Samples = 128,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten(name = 'flatten')(block2)

    dense        = Dense(nb_classes, name = 'dense',
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def GeneralModel(nb_classes, shape_spch, Chans = 64, Samples = 5*64,
            dropoutRate = 0.5, kernLength = 64, F1 = 8,
            D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
            units_lstm=32, filters_cnn_eeg=16, filters_cnn_env=16,
            units_hidden=128,
            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
            spatial_filters_mel=8, fun_act='tanh'):
    
    
    ###########
    # EEG

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    inputEEG   = Input(shape = (Samples, Chans))
    inEEG = tf.keras.layers.Reshape((Chans, Samples, 1))(inputEEG)

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(inEEG)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4*2.5))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8*2.5))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten()(block2)

    #dense        = Dense(1, name = 'dense',
    #                     kernel_constraint = max_norm(norm_rate))(flatten)
    #softmax      = Activation('softmax', name = 'softmax')(dense)
    eeg_out = flatten


    ###############################################3
    # MEL

    input_spch1 = tf.keras.layers.Input(shape=shape_spch)
    input_spch2 = tf.keras.layers.Input(shape=shape_spch)
    input_spch3 = tf.keras.layers.Input(shape=shape_spch)
    input_spch4 = tf.keras.layers.Input(shape=shape_spch)
    input_spch5 = tf.keras.layers.Input(shape=shape_spch)

    spch1_proj = input_spch1
    spch2_proj = input_spch2
    spch3_proj = input_spch3
    spch4_proj = input_spch4
    spch5_proj = input_spch5

    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer(spch1_proj)
    output_spch2 = BN_layer(spch2_proj)
    output_spch3 = BN_layer(spch3_proj)
    output_spch4 = BN_layer(spch4_proj)
    output_spch5 = BN_layer(spch5_proj)

    env_spatial_layer = tf.keras.layers.Conv1D(spatial_filters_mel, kernel_size=1)
    output_spch1 = env_spatial_layer(output_spch1)
    output_spch2 = env_spatial_layer(output_spch2)
    output_spch3 = env_spatial_layer(output_spch3)
    output_spch4 = env_spatial_layer(output_spch4)
    output_spch5 = env_spatial_layer(output_spch5)

    # layer
    BN_layer1 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer1(output_spch1)
    output_spch2 = BN_layer1(output_spch2)
    output_spch3 = BN_layer1(output_spch3)
    output_spch4 = BN_layer1(output_spch4)
    output_spch5 = BN_layer1(output_spch5)

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))

    output_spch1 = layer_exp1(output_spch1)
    output_spch2 = layer_exp1(output_spch2)
    output_spch3 = layer_exp1(output_spch3)
    output_spch4 = layer_exp1(output_spch4)
    output_spch5 = layer_exp1(output_spch5)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    output_spch1 = conv_env_layer(output_spch1)
    output_spch2 = conv_env_layer(output_spch2)
    output_spch3 = conv_env_layer(output_spch3)
    output_spch4 = conv_env_layer(output_spch4)
    output_spch5 = conv_env_layer(output_spch5)

    # layer
    BN_layer2 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer2(output_spch1)
    output_spch2 = BN_layer2(output_spch2)
    output_spch3 = BN_layer2(output_spch3)
    output_spch4 = BN_layer2(output_spch4)
    output_spch5 = BN_layer2(output_spch5)

    layer_permute = tf.keras.layers.Permute((1, 3, 2))

    output_spch1 = layer_permute(output_spch1)
    output_spch2 = layer_permute(output_spch2)
    output_spch3 = layer_permute(output_spch3)
    output_spch4 = layer_permute(output_spch4)
    output_spch5 = layer_permute(output_spch5)
    

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_spch1)[1],
                                             tf.keras.backend.int_shape(output_spch1)[2] *
                                             tf.keras.backend.int_shape(output_spch1)[3]))
    output_spch1 = layer_reshape(output_spch1)  # size = (210,32)
    output_spch2 = layer_reshape(output_spch2)
    output_spch3 = layer_reshape(output_spch3) 
    output_spch4 = layer_reshape(output_spch4)
    output_spch5 = layer_reshape(output_spch5)  

    # lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    lstm_spch = tf.compat.v1.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)
    output_spch1 = lstm_spch(output_spch1)
    output_spch2 = lstm_spch(output_spch2)
    output_spch3 = lstm_spch(output_spch3)
    output_spch4 = lstm_spch(output_spch4)
    output_spch5 = lstm_spch(output_spch5)

    stimuli_proj = [output_spch1, output_spch2, output_spch3, output_spch4, output_spch5]

    ##############################3
    # Cos
    stimuli_proj = [tf.keras.layers.Flatten()(cos_i) for cos_i in stimuli_proj]

    stimuli_proj = [tf.keras.layers.Dense(16, activation="linear")(cos_i) for cos_i in stimuli_proj]

    cos = [tf.keras.layers.Dot(1, normalize=True)([eeg_out, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear")

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]


    # Classification
    all_inputs = [inputEEG, input_spch1, input_spch2, input_spch3, input_spch4, input_spch5]
    out = tf.keras.activations.softmax((tf.keras.layers.Concatenate()(cos_proj)))

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
    print(model.summary())

    return model