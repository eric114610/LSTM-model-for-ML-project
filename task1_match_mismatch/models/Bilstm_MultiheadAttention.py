import tensorflow as tf
from tensorflow.keras import backend as K
from keras_multi_head import MultiHead

class DotLayer(tf.keras.layers.Layer):
    """ Return cosine similarity between two columns of two matrices. """

    def __init__(self):
        super(DotLayer, self).__init__()

    def call(self, list_tensors):
        layer = tf.keras.layers.Dot(axes=[2, 2], normalize=True)
        output_dot = layer(list_tensors)
        output_diag = tf.linalg.diag_part(output_dot)
        return output_diag

class Attention(tf.keras.layers.Layer):
    
    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(Attention,self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences 
        })
        return config
    

def lstm_mel(shape_eeg, shape_spch, units_lstm=64, filters_cnn_eeg=16, filters_cnn_env=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
                            spatial_filters_mel=8, fun_act='tanh'):
    """
    Return an LSTM based model where batch normalization is applied to input of each layer.

    :param shape_eeg: a numpy array, shape of EEG signal (time, channel)
    :param shape_spch: a numpy array, shape of speech signal (time, feature_dim)
    :param units_lstm: an int, number of units in LSTM
    :param filters_cnn_eeg: an int, number of CNN filters applied on EEG
    :param filters_cnn_env: an int, number of CNN filters applied on envelope
    :param units_hidden: an int, number of units in the first time_distributed layer
    :param stride_temporal: an int, amount of stride in the temporal direction
    :param kerSize_temporal: an int, size of CNN filter kernel in the temporal direction
    :param fun_act: activation function used in layers
    :return: LSTM-based model
    """
    
    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    input_spch1 = tf.keras.layers.Input(shape=shape_spch)
    input_spch2 = tf.keras.layers.Input(shape=shape_spch)
    input_spch3 = tf.keras.layers.Input(shape=shape_spch)
    input_spch4 = tf.keras.layers.Input(shape=shape_spch)
    input_spch5 = tf.keras.layers.Input(shape=shape_spch)


    ############
    #### upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1), strides=(stride_temporal, 1), activation="relu")(output_eeg)

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)

    output_eeg = tf.keras.layers.Dense(104, activation="relu")(output_eeg)
    output_eeg = tf.keras.layers.Dense(104, activation="sigmoid")(output_eeg)


    ##############
    #### Bottom part of the network dealing with Speech.

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

    output_spch1 = layer_exp1(output_spch1)
    output_spch2 = layer_exp1(output_spch2)
    output_spch3 = layer_exp1(output_spch3)
    output_spch4 = layer_exp1(output_spch4)
    output_spch5 = layer_exp1(output_spch5)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1), strides=(stride_temporal, 1), activation="relu")
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

    lstm_spch = tf.compat.v1.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)
    BiLstm_spch = tf.keras.layers.Bidirectional(lstm_spch)
    output_spch1 = BiLstm_spch(output_spch1)
    output_spch2 = BiLstm_spch(output_spch2)
    output_spch3 = BiLstm_spch(output_spch3)
    output_spch4 = BiLstm_spch(output_spch4)
    output_spch5 = BiLstm_spch(output_spch5)

    att_layer = Attention(return_sequences=True)

    output_spch1 = MultiHead(att_layer, layer_num=4, name='Multihead_Att1')(output_spch1)
    output_spch2 = MultiHead(att_layer, layer_num=4, name='Multihead_Att2')(output_spch2)
    output_spch3 = MultiHead(att_layer, layer_num=4, name='Multihead_Att3')(output_spch3)
    output_spch4 = MultiHead(att_layer, layer_num=4, name='Multihead_Att4')(output_spch4)
    output_spch5 = MultiHead(att_layer, layer_num=4, name='Multihead_Att5')(output_spch5)

    shape = output_spch1.get_shape().as_list()  

    output_spch1 = tf.reshape(output_spch1, [-1, shape[1] , shape[2] * shape[3]])
    output_spch2 = tf.reshape(output_spch2, [-1, shape[1] , shape[2] * shape[3]])
    output_spch3 = tf.reshape(output_spch3, [-1, shape[1] , shape[2] * shape[3]])
    output_spch4 = tf.reshape(output_spch4, [-1, shape[1] , shape[2] * shape[3]])
    output_spch5 = tf.reshape(output_spch5, [-1, shape[1] , shape[2] * shape[3]])

    lstm_spch2 = tf.compat.v1.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)

    output_spch1 = lstm_spch2(output_spch1)
    output_spch2 = lstm_spch2(output_spch2)
    output_spch3 = lstm_spch2(output_spch3)
    output_spch4 = lstm_spch2(output_spch4)
    output_spch5 = lstm_spch2(output_spch5)

    output_spch1 = tf.keras.layers.Dense(104, activation="relu")(output_spch1)
    output_spch1 = tf.keras.layers.Dense(104, activation="sigmoid")(output_spch1)
    output_spch2 = tf.keras.layers.Dense(104, activation="relu")(output_spch2)
    output_spch2 = tf.keras.layers.Dense(104, activation="sigmoid")(output_spch2)
    output_spch3 = tf.keras.layers.Dense(104, activation="relu")(output_spch3)
    output_spch3 = tf.keras.layers.Dense(104, activation="sigmoid")(output_spch3)
    output_spch4 = tf.keras.layers.Dense(104, activation="relu")(output_spch4)
    output_spch4 = tf.keras.layers.Dense(104, activation="sigmoid")(output_spch4)
    output_spch5 = tf.keras.layers.Dense(104, activation="relu")(output_spch5)
    output_spch5 = tf.keras.layers.Dense(104, activation="sigmoid")(output_spch5)


    ##############
    #### last common layers
    # layer
    layer_dot = DotLayer()
    cos_scores = layer_dot([output_eeg, output_spch1])
    cos_scores2 = layer_dot([output_eeg, output_spch2])
    cos_scores3 = layer_dot([output_eeg, output_spch3])
    cos_scores4 = layer_dot([output_eeg, output_spch4])
    cos_scores5 = layer_dot([output_eeg, output_spch5])

    cos = [cos_scores, cos_scores2, cos_scores3, cos_scores4, cos_scores5]
    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear")
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]
    out = tf.keras.activations.softmax((tf.keras.layers.Concatenate()(cos_proj)))


    model = tf.keras.Model(inputs=[input_eeg, input_spch1, input_spch2, input_spch3, input_spch4, input_spch5], outputs=[out])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
    )
    print(model.summary())

    return model

