import time
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from data_preprocess import get_boject_six_matrix
from keras import backend as kb
import tensorflow as tf
import os
import logging
import gc
from keras.utils import multi_gpu_model
from models_generator import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r'

class TextClassifier():


    def train(self, x_train, label_train, input_length, word_size, y_num):
        self.oof_nn_model_two_gpu(x_train, label_train, input_length, word_size, y_num)


    def oof_nn_model_two_gpu(self, x_train,label_train, input_length, word_size, y_num):

        print("train ===============> ")
        """
        x_train_padded_seqs, x_test_padded_seqs, y_train_one_hot, y_test_one_hot = train_test_split(x_train,
                                                                                                    label_train,
                                                                                                    test_size=0.1,
                                                                                                    random_state=666)
        """
        if not os.path.exists("object_six_multilabel_classifier_model"):
            os.mkdir("object_six_multilabel_classifier_model")
        check_point = ModelCheckpoint(filepath="./object_six_multilabel_classifier_model/" + time.strftime("%Y%m%d%H%M%S",
                                                                                       time.localtime()) + '##epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5',
                                      monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                      mode='auto', period=1)

        reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto',
                                                     epsilon=0.0001, cooldown=0, min_lr=0)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

        tensorboard = keras.callbacks.TensorBoard(log_dir='../logs/BiLSTM', histogram_freq=0, write_graph=True,
                                                  write_images=False,
                                                  embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        self.classifier = BiLSTM(input_size=word_size, output_size=y_num, input_length=input_length)
        parallel_model = multi_gpu_model(self.classifier, gpus=2)
        parallel_model.summary()
        parallel_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        keras.utils.plot_model(parallel_model, to_file='save/model.png')  # Save a graphical representation of the model

        parallel_model.fit(x_train, label_train,
                  batch_size=512,
                  epochs=100,
                  validation_split=0.1,
                 # validation_data=(x_test_padded_seqs, y_test_one_hot),
                  callbacks=[check_point, reducelr, earlystop, tensorboard])
        del self.classifier
        del parallel_model
        gc.collect()
        kb.clear_session()
        tf.reset_default_graph()


if __name__ == '__main__':
    length = 300
    content_list_seq_pad, label_list, length_vocab, length_label, column_num = get_boject_six_matrix(input_length=length, path="./dataset/train_data_after_cut.xlsx")
    text_classifier = TextClassifier()
    logger.info("start train model")
    text_classifier.train(content_list_seq_pad, label_list, length, length_vocab, column_num)
    logger.info("complete train model")
    del text_classifier