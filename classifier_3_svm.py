from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from data_preprocess import get_train_data_3
import logging
import os
import gc
import subprocess
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


if __name__ == '__main__':

    # load train data
    logger.info("start load data")
    data_dict = get_train_data_3(path="./dataset/data_reform/train_reform_content_after_cut.csv")
    for key in data_dict.keys():
        content_train = data_dict[key][0]
        label_train = np.array(data_dict[key][1])
        logger.info("start train %s feature extraction" % column_list[column_list.index(key)])
        vectorizer_tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_words, lowercase=True,
                                           ngram_range=(1, 5), min_df=5, norm='l2')
        features_train = vectorizer_tfidf.fit_transform(content_train)
        logger.info("complete train %s feature extraction" % column_list[column_list.index(key)])
        print(column_list[column_list.index(key)] + " " + "trainset vocab shape: " + str(features_train.shape))

        #data format
        logger.info("start data format %s" % column_list[column_list.index(key)])
        train_features_save_path = "./classifier_3_train_features_svm_format_files/" + column_list[column_list.index(key)]
        if not os.path.exists(train_features_save_path):
            os.makedirs(train_features_save_path)
        dump_svmlight_file(features_train, label_train, train_features_save_path + '/' + column_list[column_list.index(key)]
                           + '.txt', zero_based=True, comment=None, query_id=None)
        logger.info("complete data format %s" % column_list[column_list.index(key)])
    logger.info("complete all data format")

    logger.info("start train model")
    for column in column_list:
        model_save_path = "./model_files_classifier_3/" + column
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        subprocess.call("./thundersvm-master/build/bin/thundersvm-train -c 100 -g 0.5 " +
                        "./classifier_3_train_features_svm_format_files/" + column + "/" + column + ".txt "
                        + model_save_path + "/" + column + ".model", shell=True)
        logger.info("complete train %s model" % column)
        logger.info("complete save %s model" % column)
        gc.collect()
        KTF.clear_session()
        tf.reset_default_graph()
    logger.info("complete train all models")
    logger.info("complete save all models")
    subprocess.call("python evaluate.py", shell=True)

