import time
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from data_preprocess import get_train_data, get_segment_train_data, get_multilabel_train_data, get_segment_train_data2
from keras import backend as kb
import tensorflow as tf
import keras
import os
import logging
import gc
from keras.utils import multi_gpu_model
from models_generator import *
import pandas as pd
import json
import jieba
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier():

   # def __init__(self, word_embedding):
       # self.word_embedding = word_embedding

    def train(self, model_name, x_train, y_train, x_val, y_val, input_length, word_size, y_num):
        self.oof_nn_model_two_gpu(model_name, x_train, y_train, x_val, y_val, input_length, word_size, y_num)

    def oof_nn_model_two_gpu(self, model_name, x_train, y_train, x_val, y_val, input_length, word_size, y_num):

        print("train ===============> ")
        #x_train_padded_seqs, x_test_padded_seqs, y_train_one_hot, y_test_one_hot = train_test_split(x_val,
                                                                                                    #y_val,
                                                                                                    #test_size=1,
                                                                                                   # shuffle=True,
                                                                                                  # random_state=2580)
        x_train_padded_seqs2, x_test, y_train_one_hot2, y_test = train_test_split(x_train,
                                                                                  y_train,
                                                                                  test_size=0,
                                                                                  shuffle=True,
                                                                                  random_state=2580)
        if not os.path.exists(model_name):
            os.chdir("./model_files_bigru_attention")
            os.makedirs(model_name)
            os.chdir("../")
        check_point = ModelCheckpoint(filepath="./model_files_bigru_attention" + '/' + model_name +
                                               '/' + time.strftime("%Y%m%d%H%M%S", time.localtime()) +
                                               '##epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5',
                                      monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                      mode='auto', period=1)

        #reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto',
         #                                            min_delta=0.0001, cooldown=0, min_lr=0)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

        tensorboard = keras.callbacks.TensorBoard(log_dir='../logs/model', histogram_freq=0, write_graph=True,
                                                  write_images=False,
                                                  embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        self.classifier = model(input_size=word_size, output_size=y_num, input_length=input_length)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        parallel_model = multi_gpu_model(self.classifier, gpus=2)
        parallel_model.summary()
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer=adam,
                               metrics=['accuracy'])

        parallel_model.fit(x_train_padded_seqs2, y_train_one_hot2,
                           batch_size=512,
                           epochs=15,
                          #validation_split=0.1,
                           validation_data=(x_val, y_val),
                           shuffle=True,
                           callbacks=[check_point, earlystop, tensorboard])
        del self.classifier
        del parallel_model
        gc.collect()
        kb.clear_session()
        tf.reset_default_graph()

column_list = [
     "location_traffic_convenience",
     "location_distance_from_business_district",
     "location_easy_to_find",
     "service_wait_time",
     "service_waiters_attitude",
     "service_parking_convenience",
     "service_serving_speed",
     "price_level",
     "price_cost_effective",
     "price_discount",
     "environment_decoration",
     "environment_noise",
     "environment_space",
     "environment_cleaness",
     "dish_portion",
     "dish_taste",
     "dish_look",
     "dish_recommendation",
     "others_overall_experience",
     "others_willing_to_consume_again"
]

if __name__ == '__main__':

    MAX_SEQUENCE_LENGTH = 1000
    #MAX_NUM_WORDS = 200000
    segment_data_dic, y_num = get_segment_train_data(input_length=MAX_SEQUENCE_LENGTH,
                                                     path="./dataset/data_reform/train_reform_content_after_cut.csv")
    segment_data_dic2, y_num2 = get_segment_train_data2(input_length=MAX_SEQUENCE_LENGTH,
                                                        path="./dataset/data_reform/train_reform_content_after_cut.csv")
    df_val, x_val, y_num_val, len_vocab = get_multilabel_train_data(input_length=MAX_SEQUENCE_LENGTH,
                                                                    path="./dataset/valid.csv")
    for column in column_list:
        key = segment_data_dic[column]
        train_content = tuple(key[0])
        train_label = key[1]
        #print(train_label)
        label_val = df_val[column]
        label_list_onehot = []
        for data in label_val:

            label = [0, 0, 0, 0]

            if data == 1:
                label[0] = 1
            if data == 0:
                label[1] = 1
            if data == -1:
                label[2] = 1
            if data == -2:
                label[3] = 1
            label_list_onehot.append(label)
        y_val = np.array(label_list_onehot)

        key2 = segment_data_dic2[column]
        train_content2 = tuple(key2[0])
        train_label2 = key2[1]

        train_content_matrix = np.array(train_content)
        train_content_matrix2 = np.array(train_content2)

        train_label_matrix = np.array(train_label)
        train_label2_matrix = np.array(train_label2)

        shape = train_label2_matrix.shape #return type:"tuple"
        row_num = shape[0]
        extract_data_rate = 1
        row_num_update = int(extract_data_rate * row_num)
        train_label2_matrix_update = train_label2_matrix[0:row_num_update, ...]

        train_label_matrix_concatenate = np.concatenate((train_label_matrix, train_label2_matrix_update))

        content_list_seq_pad2_update = train_content_matrix2[0:row_num_update, ...]
        content_list_seq_pad_concatenate = np.concatenate((train_content_matrix, content_list_seq_pad2_update))
        #print(train_label_matrix)
       # print(train_content_matrix.shape)
        #print(train_content_matrix.ndim)
        #print('\n')
        #print(train_label_matrix.shape)
        #print(train_label_matrix.ndim)

        count1 = 0
        for item in train_label_matrix_concatenate:

            idx = np.argmax(item)
            if idx == 0:
                count1 = count1 + 1
        print(column + " number of 1 :" + str(count1))

        count2 = 0
        for item in train_label_matrix_concatenate:

            idx = np.argmax(item)
            if idx == 1:
                count2 = count2 + 1
        print(column + " number of 0 :" + str(count2))

        count3 = 0
        for item in train_label_matrix_concatenate:

            idx = np.argmax(item)
            if idx == 2:
                count3 = count3 + 1
        print(column + " number of -1 :" + str(count3))

        count4 = 0
        for item in train_label_matrix_concatenate:

            idx = np.argmax(item)
            if idx == 3:
                count4 = count4 + 1
        print(column + " number of -2 :" + str(count4))

        #if os.path.exists("vocab.json"):
         #   with open("vocab.json", encoding="utf-8") as f:
         #       vocab = json.load(f)
          #      len_vocab = len(vocab)
       # word_embedding_matrix = pd.read_pickle('./gensim_data_word_embedding_matrix_new.pkl')
        text_classifier = TextClassifier()
        logger.info("start train  %s model" % column)
        #text_classifier.train(column, content_list_seq_pad_concatenate, train_label_matrix_concatenate, length, len_vocab, y_num)
        text_classifier.train(column, content_list_seq_pad_concatenate, train_label_matrix_concatenate, x_val, y_val, MAX_SEQUENCE_LENGTH,
                              len_vocab, y_num)
        logger.info("complete train %s model" % column)
        del text_classifier



