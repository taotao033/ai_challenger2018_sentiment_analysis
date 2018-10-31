import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras import backend
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import *
from keras.layers import *
from att_layer import AttentionWeightedAverage
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPool1D, Conv1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import BatchNormalization
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras.engine.topology import Layer
import keras
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier():

    def __init__(self, vectorizer, classifier=MultinomialNB()):
        classifier = SVC(kernel="rbf")
        # classifier = SVC(kernel="linear")
        self.classifier = classifier
        self.vectorizer = vectorizer

    def features(self, x):
        return self.vectorizer.transform(x)

    def fit(self, x, y):

        self.classifier.fit(self.features(x), y)

    def predict(self, x):

        return self.classifier.predict(self.features(x))

    def score(self, x, y):
        return self.classifier.score(self.features(x), y)

    def get_f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='macro')



class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # (x, 40, 1)
        uit = K.squeeze(uit, -1)  # (x, 40)
        uit = uit + self.b  # (x, 40) + (40,)
        uit = K.tanh(uit)  # (x, 40)

        ait = uit * self.u  # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait)  # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            ait = mask * ait  # (x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[-1])

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        #self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)

        features_dim = self.features_dim
        # step_dim = self.step_dim
        step_dim = input_shape[1]

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b[:input_shape[1]]

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
    	return input_shape[0], self.features_dim
# end Attention


def model(input_size, output_size, input_length=20):
    main_input = Input(shape=(input_length,))
    encode = Bidirectional(CuDNNGRU(128, return_sequences=True))
    encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))
    attention = Attention(input_length)
    embed = Embedding(input_size + 1, 256, input_length=input_length)(main_input)
    x_3 = SpatialDropout1D(0.2)(embed)
    x_3 = encode(x_3)
    x_3 = Dropout(0.5)(x_3)
    x_3 = encode2(x_3)
    x_3 = Dropout(0.5)(x_3)
    avg_pool_3 = GlobalAveragePooling1D()(x_3)
    max_pool_3 = GlobalMaxPooling1D()(x_3)
    attention_3 = attention(x_3)
    x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
    main_ouput = Dense(output_size, activation="sigmoid")(x)
    model = Model(inputs=main_input, outputs=main_ouput)
   # model.compile(
        #loss='categorical_crossentropy',
        #optimizer=adam)
    return model

def build_model(word_embedding_matrix, input_size, output_size, input_length=20):

    embedding_layer = Embedding(input_size + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                input_length=input_length, trainable=False)

    main_input = Input(shape=(input_length,), dtype='float64')
    embed = embedding_layer(main_input)
    x = SpatialDropout1D(0.2)(embed)
    rnn = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    cnn1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(rnn)
    cnn2 = Conv1D(filters=64, kernel_size=4, activation='relu', padding='same')(rnn)
    cnn3 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(rnn)
    x = concatenate([rnn, cnn1, cnn2, cnn3])
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    x = concatenate([avg_pool, max_pool, attn])

    x = Dropout(0.8)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='elu')(x)

    x = BatchNormalization()(x)

    preds = Dense(output_size, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=preds)

    return model



def CNN(input_size, output_size,input_length = 20):
    model = Sequential()
    model.add(Embedding(input_size, 256, input_length=input_length))
    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(256, 3, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(128, 3, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    return model


def CNN_GRU(input_size, output_size, input_length=20):
    """
    embedding_layer = Embedding(input_size + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                input_length=input_length, trainable=False)

    main_input = Input(shape=(input_length,), dtype='float64')
    embed = embedding_layer(main_input)
    x = Convolution1D(1024, 3, padding='same', strides=1, data_format=None, activation='selu')(embed)
    x = MaxPool1D(pool_size=2)(x)
    x = GRU(512, dropout=0.7, recurrent_dropout=0.5, return_sequences=True)(x)
    x = GRU(256, dropout=0.7, recurrent_dropout=0.5)(x)
    main_output = Dense(output_size, activation='softmax')(x)
    model = Model(inputs=main_input, outputs=main_output)
    """
    model = Sequential()
    model.add(Embedding(input_size, 512, input_length=input_length))
    model.add(Convolution1D(1024, 3, padding='same', strides=1, data_format=None))
    model.add(Activation('selu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GRU(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(GRU(256, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(output_size, activation='sigmoid'))

    return model


def CNN_LSTM(input_size, output_size, input_length=20):
    main_input = Input(shape=(input_length,), dtype='float64')
    embed = Embedding(input_size, 256, input_length=input_length)(main_input)
    cnn = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPool1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.5))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn, rnn], axis=-1)
    main_output = Dense(output_size, activation='softmax')(con)
    model = Model(inputs=main_input, outputs=main_output)
    return model

def BiLSTM(input_size, output_size,input_length = 20):
    """
    MAX_NB_WORDS = 200000

    #MAX_SEQUENCE_LENGTH = 1000  # 893


    main_input = Input(shape=(input_length,), dtype='float64')
    embedding_layer = Embedding(MAX_NB_WORDS+1,
                                word_embedding_matrix.shape[1],
                                weights=[word_embedding_matrix],
                                input_length=input_length,
                                trainable=False)
    embedded_squences = embedding_layer(main_input)
    #main_input = Input(shape=(input_length,), dtype='float64')(embedding_layer)
    #embed = Embedding(input_size, 256, input_length=input_length)(embedded_squences)
    rnn = Bidirectional(LSTM(128, return_sequences=True))(embedded_squences)
    cnn = Convolution1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(rnn)
    cnn = GlobalMaxPool1D()(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Dense(50, activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    #con = concatenate([rnn, cnn], axis=-1)
    main_output = Dense(output_size, activation='sigmoid')(cnn)
    model = Model(inputs=main_input, outputs=main_output)
    """
    model = Sequential()
    model.add(Embedding(input_size, 256, input_length=input_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform"))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    return model

def FAST_TEXT(input_size, output_size, n_value=2, x_train_padded_seqs=None, x_train_word_ids=None,
              x_test_word_ids=None,maxlen=64):
    def create_ngram_set(input_list, ngram_value=n_value):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    # Add the new n-gram generated words into the original sentence sequence
    def add_ngram(sequences, token_indice, ngram_range=n_value):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    ngram_set = set()
    for input_list in x_train_padded_seqs:
        for i in range(2, n_value + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)
    start_index = input_size + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # 给bigram词汇编码
    indice_token = {token_indice[k]: k for k in token_indice}
    max_features = np.max(list(indice_token.keys())) + 1
    x_train = add_ngram(x_train_word_ids, token_indice, 3)
    x_test = add_ngram(x_test_word_ids, token_indice, 3)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 256, input_length=maxlen))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(output_size, activation='sigmoid'))
    return model, x_train, x_test


def GRU_ATTENTION(input_size, output_size,input_length = 20):
    inputs = Input(shape=(input_length,), dtype='float64')
    embed = Embedding(input_size, 512, input_length=input_length)(inputs)
    gru = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embed)
    attention = AttLayer()(gru)
    output = Dense(output_size, activation='softmax')(attention)
    model = Model(inputs, output)
    return model


def SIMPLE_GRU(input_size, output_size,input_length = 20):
    model = Sequential()
    model.add(Embedding(input_size, 256, input_length=input_length))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    return model


def SIMPLE_LSTM(input_size, output_size,input_length = 20):
    model = Sequential()
    model.add(Embedding(input_size, 256, input_length=input_length))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    return model


def TEXT_CNN(input_size, output_size,input_length = 20):
    main_input = Input(shape=(input_length,), dtype='float64')
    embedder = Embedding(input_size, 300, input_length=input_length)
    embed = embedder(main_input)
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(output_size, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    return model


def TEXT_CNN_BI_GRU(input_size, output_size,input_length = 20):
    #main_input = Input(shape=(input_length,), dtype='float64')
    #embed = Embedding(input_size,512 , input_length=input_length)(main_input)
    #cnn1 = Convolution1D(512, 3, padding='same', strides=1, activation='relu')(embed)
    #cnn1 = MaxPool1D(pool_size=4)(cnn1)
    #cnn2 = Convolution1D(512, 4, padding='same', strides=1, activation='relu')(embed)
    #cnn2 = MaxPool1D(pool_size=4)(cnn2)
    #cnn3 = Convolution1D(512, 5, padding='same', strides=1, activation='relu')(embed)
    #cnn3 = MaxPool1D(pool_size=4)(cnn3)
    #cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    #gru = Bidirectional(GRU(256, dropout=0.5, recurrent_dropout=0.1))(cnn)
    #main_output = Dense(output_size, activation='sigmoid')(gru)
    #model = Model(inputs=main_input, outputs=main_output)

    main_input = Input(shape=(input_length,), dtype='float64')
    embed = Embedding(input_size,512 , input_length=input_length)(main_input)
    cnn1 = Convolution1D(512, 3, padding='same', strides=1)(embed)
    cnn1 = LeakyReLU()(cnn1)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(512, 4, padding='same', strides=1)(embed)
    cnn2 = LeakyReLU()(cnn2)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(512, 5, padding='same', strides=1)(embed)
    cnn3 = LeakyReLU()(cnn3)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    gru = Bidirectional(GRU(256, dropout=0.5, recurrent_dropout=0.1,activation=None))(cnn)
    main_output = Dense(output_size, activation='sigmoid')(gru)
    model = Model(inputs=main_input, outputs=main_output)
    return model


def TEXT_RCNN(input_size, x_train=None, x_test=None, y_train=None, y_test=None,maxlen=20):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    total_sentence = x_train.append(x_test)
    tokenizer.fit_on_texts(total_sentence)
    vocab = tokenizer.word_index
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=maxlen)
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=maxlen)
    left_train_word_ids = [[len(vocab)] + x[:-1] for x in x_train_word_ids]
    left_test_word_ids = [[len(vocab)] + x[:-1] for x in x_test_word_ids]
    right_train_word_ids = [x[1:] + [len(vocab)] for x in x_train_word_ids]
    right_test_word_ids = [x[1:] + [len(vocab)] for x in x_test_word_ids]
    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=maxlen)
    left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=maxlen)
    right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=maxlen)
    right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=maxlen)

    x_train_seq = [x_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs]
    x_test_seq = [x_test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs]

    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    embedder = Embedding(input_size, 300, input_length=maxlen)
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)
    forward = LSTM(256, return_sequences=True)(l_embedding)  # See equation (1)
    backward = LSTM(256, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2)
    together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3)
    semantic = TimeDistributed(Dense(128, activation="tanh"))(together)  # See equation (4)
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(128,))(semantic)  # See equation (5)
    output = Dense(10, activation="sigmoid")(pool_rnn)  # See equations (6) and (7)
    model = Model(inputs=[document, left_context, right_context], outputs=output)

    return model, x_train_seq, x_test_seq, y_train, y_test

test = CNN_LSTM(1000,20)