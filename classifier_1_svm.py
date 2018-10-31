import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.datasets import dump_svmlight_file
from keras.utils import multi_gpu_model
#from keras import backend as kb
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from data_preprocess import get_train_data
from models_generator import TextClassifier
import argparse
import logging
import jieba
import os
import gc
import subprocess
#import theano
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)

stop_words = []
with open("./dataset/哈工大停用词表.txt", "r") as f:
    for line in f.readlines():
        stop_words.append(line)

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

#
# def f1_macro(df_predict, df_real):
#
#     df_val = pd.read_csv("./dataset/valid.csv")
#     f1_per_column = open("./output/classifier1_svm_f1_score_predict_validation.txt", 'a+')
#     f1_sum = 0
#
#     for column in column_list:
#         f1 = f1_score(df_real[column], df_predict[column], average='macro')
#         f1_per_column.write(column + '-f1_score: ' + str(f1) + '\n')
#         f1_sum += f1
#     average_f1 = f1_sum / len(column_list)
#     f1_per_column.write('average_f1: ' + str(average_f1))
#     f1_per_column.close()
#     if average_f1 >= 0.750:
#         print("Great! Nice! you win!")
#     else:
#         print("Unfortunately,come on!")
#     print('f1_score: ' + str(average_f1))


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-mn', '--model_name', type=str, nargs='?',
    #                     help='the name of model')
    #
    # args = parser.parse_args()
    # model_name = args.model_name
    # if not model_name:
    #     model_name = "model_dict.pkl"
    # load train data
    logger.info("start load data")
    train_label_list = get_train_data(path="./dataset/train.csv")
    #val_label_list = get_train_data(path="./dataset/valid.csv")

    train_data_df = pd.read_csv("./dataset/data_reform/train_reform_content_after_cut.csv", encoding='utf-8')
    #validate_data_df = pd.read_csv("./dataset/valid.csv", encoding='utf-8')
    #validate_data_df2 = pd.read_csv("./dataset/valid.csv", encoding='utf-8')

    content_train = train_data_df.iloc[:, 1]
    #content_validate = validate_data_df.iloc[:, 1]

    # logger.info("start seg train data")
    # content_list = []
    # for text in content_train:
    #     token = jieba.cut(text)
    #
    #     arr_temp = []
    #     for item in token:
    #         arr_temp.append(item)
    #     content_list.append(" ".join(arr_temp))
    # content_train = content_list
    # logger.info("complete seg train data")

    # logger.info("start seg validate data")
    # content_val_list = []
    # for text in content_validate:
    #     token = jieba.cut(text)
    #
    #     arr_temp = []
    #     for item in token:
    #         arr_temp.append(item)
    #     content_val_list.append(" ".join(arr_temp))
    # content_validate = content_val_list
    # logger.info("complete seg validate data")

    logger.info("start train feature extraction")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_words, lowercase=True,
                                       ngram_range=(1, 5), min_df=5, norm='l2')
    features_train = vectorizer_tfidf.fit_transform(content_train)

    #features_val = vectorizer_tfidf.transform(content_validate)
    # dump_svmlight_file(features_val, val_label_list[..., 0], 'svm_output_val.txt', zero_based=True, comment=None,
    #                    query_id=None)
    logger.info("complete train feature extraction")
    print("trainset vocab shape: " + str(features_train.shape))
    #print("validationset vocab shape: " + str(features_val.shape))

    #data format
    logger.info("start format data ")
    #classifier_dict = dict()
    for column in column_list:
        label_train = train_label_list[..., column_list.index(column)]
        logger.info("start format %s" % column)
        train_features_save_path = "./classifier_1_train_features_svm_format_files/" + column
        if not os.path.exists(train_features_save_path):
            os.makedirs(train_features_save_path)
        dump_svmlight_file(features_train, label_train, train_features_save_path + '/' + column + '.txt',
                           zero_based=True, comment=None, query_id=None)
    logger.info("complete format data")

    #model train
    logger.info("start train model")
    for column in column_list:
        model_save_path = "./model_files_classifier_1/" + column
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        subprocess.call("./thundersvm-master/build/bin/thundersvm-train -c 100 -g 0.5 " +
                        "./classifier_1_train_features_svm_format_files/" + column + "/" + column + ".txt "
                        + model_save_path + "/" + column + ".model", shell=True)
        logger.info("complete train %s model" % column)
        logger.info("complete save %s model" % column)
        gc.collect()
        KTF.clear_session()
        tf.reset_default_graph()
    logger.info("complete train all models")
    logger.info("complete save all models")
    subprocess.call("python classifier_2_svm.py", shell=True)
    #logger.info("start validate model")
    # validate model
    """
    #f1_score_dict = dict()
    for column in column_list:
        label_validate_real = val_label_list[..., column_list.index(column)]
        text_classifier = classifier_dict[column]
        predictions = text_classifier.predict(features_val)
        validate_data_df[column] = predictions
        validate_data_df2[column] = label_validate_real
    f1_macro(validate_data_df, validate_data_df2)

    logger.info("complete validate model")

    # save model
    logger.info("start save model")
    model_save_path = "./model_files_classifier1_svm/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    joblib.dump(classifier_dict, model_save_path + model_name)
    logger.info("complete save model")
    """
