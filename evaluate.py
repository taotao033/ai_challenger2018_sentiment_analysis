import pandas as pd
import logging
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
import numpy as np
import os
import subprocess
from data_preprocess import get_train_data_2, get_train_data_3
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


stop_words = []

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

    # #stage 1 start
    validation_data_df = pd.read_csv("./dataset/valid.csv", encoding='utf-8')
    content_validation = validation_data_df.iloc[:, 1]

    logger.info("start seg validate data")
    content_val_list = []
    for text in content_validation:
        token = jieba.cut(text)

        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_val_list.append(" ".join(arr_temp))
    content_validation = content_val_list
    logger.info("complete seg validate data")

    logger.info("start validation feature extraction")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_words, lowercase=True,
                                       ngram_range=(1, 5), min_df=5, norm='l2')
    train_data_df = pd.read_csv("./dataset/data_reform/train_reform_content_after_cut.csv", encoding='utf-8')
    content_train = train_data_df.iloc[:, 1]
    features_train = vectorizer_tfidf.fit_transform(content_train)
    features_val =vectorizer_tfidf.transform(content_validation)  # return type:<class 'scipy.sparse.csr.csr_matrix'>
    # start format data
    logger.info("start data format")
    for column in column_list:
        label_validation = validation_data_df[column]
        label_validation = np.array(label_validation)
        save_path = "./predict_samples_classifier_1_val_features_svm_format_files/" + column
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dump_svmlight_file(features_val, label_validation, save_path + "/" + column + ".txt",
                           zero_based=True, comment=None, query_id=None)
    logger.info("complete data format ")
    #start load model and predict label
    logger.info("start load model and predict label")
    for column in column_list:
        predict_result_save_path = "./predict_results_classifier_1/" + column
        if not os.path.exists(predict_result_save_path):
            os.makedirs(predict_result_save_path)
        subprocess.call("./thundersvm-master/build/bin/thundersvm-predict " +
                        "./predict_samples_classifier_1_val_features_svm_format_files/" + column + "/" + column + ".txt "
                        + "./model_files_classifier_1/" + column + "/" + column + ".model " +
                        predict_result_save_path + "/" + column + "_predict_result.csv", shell=True)
        #add .csv header, to make the next step easier.
        with open(predict_result_save_path + "/" + column + "_predict_result.csv", "r", encoding="utf-8") as f:
            temp = []
            for line in f.readlines():
                temp.append(line[0])
            data_frame = pd.DataFrame({column: temp})
            data_frame.to_csv(predict_result_save_path + "/" + column + "_predict_result.csv", index=False, sep=",",
                              encoding="utf-8")
        #overwrite the original file,complete add .csv header

        df = pd.read_csv(predict_result_save_path + "/" + column + "_predict_result.csv", encoding="utf-8")
        label_new = []
        label = df[column]
        for la in label:
            if la == 0:
                label_new.append(-2)
            else:
                label_new.append(1)
        validation_data_df[column] = label_new
    validation_data_df.to_csv("./predict_results_classifier_1/validation_predict.csv", index=False, sep=",", encoding="utf-8")
    print("complete 20 labels binary classification predict.The result is saved under the path: ./predict_results_classifier_1/validation_predict.csv")
    #stage 1 end

    #stage 2 start
    data_dict = get_train_data_2("./dataset/data_reform/train_reform_content_after_cut.csv")
    validation_data_df2 = pd.read_csv("./predict_results_classifier_1/validation_predict.csv", encoding="utf-8")
    for key in data_dict.keys():
        content_validation2 = validation_data_df2[validation_data_df2[column_list[column_list.index(key)]] == 1]["content"]
        label_validation2 = validation_data_df2[validation_data_df2[column_list[column_list.index(key)]] == 1][key]
        print(content_validation2.shape)
        print(label_validation2.shape)
        features_train2 = vectorizer_tfidf.fit_transform(data_dict[key][0])
        features_val2 = vectorizer_tfidf.transform(content_validation2)

        # start format data
        logger.info("start %s data format " % key)
        label_validation2 = np.array(label_validation2)
        save_path = "./predict_samples_classifier_2_val_features_svm_format_files/" + key
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dump_svmlight_file(features_val2, label_validation2, save_path + "/" + key + ".txt",
                           zero_based=True, comment=None, query_id=None)
        logger.info("complete %s data format" % key)
    logger.info("complete all data format")

    #start load model and predict label
    logger.info("start load model and predict label")
    for column in column_list:
        predict_result_save_path2 = "./predict_results_classifier_2/" + column
        if not os.path.exists(predict_result_save_path2):
            os.makedirs(predict_result_save_path2)
        subprocess.call("./thundersvm-master/build/bin/thundersvm-predict " +
                        "./predict_samples_classifier_2_val_features_svm_format_files/" + column + "/" + column + ".txt "
                        + "./model_files_classifier_2/" + column + "/" + column + ".model " +
                        predict_result_save_path2 + "/" + column + "_predict_result.csv", shell=True)
        #add .csv header, to make the next step easier.
        with open(predict_result_save_path2 + "/" + column + "_predict_result.csv", "r", encoding="utf-8") as f:
            temp = []
            for line in f.readlines():
                temp.append(line[0])
            data_frame = pd.DataFrame({column: temp})
            data_frame.to_csv(predict_result_save_path2 + "/" + column + "_predict_result.csv", index=False, sep=",",
                              encoding="utf-8")
        #overwrite the original file,complete add .csv header

        df = pd.read_csv(predict_result_save_path2 + "/" + column + "_predict_result.csv", encoding="utf-8")
        label_new = []
        label = df[column]
        for la in label:
            if la == 1:
                label_new.append(1)
            else:
                label_new.append(0)

        val_label = validation_data_df2[column]
        count = 0
        label_new2 = []
        for li in val_label:
            if li == -2:
                label_new2.append(-2)
            else:
                label_new2.append(label_new[count])
                count = count + 1
        validation_data_df2[column] = label_new2
    validation_data_df2.to_csv("./predict_results_classifier_2/validation_predict.csv", index=False, sep=",", encoding="utf-8")
    print("complete 20 labels binary classification predict.The result is saved under the path: ./predict_results_classifier_2/validation_predict.csv")
    #stage 2 end

    #stage 3 start
    data_dict3 = get_train_data_3("./dataset/data_reform/train_reform_content_after_cut.csv")
    validation_data_df3 = pd.read_csv("./predict_results_classifier_2/validation_predict.csv", encoding="utf-8")
    for key in data_dict3.keys():

        content_validation3 = validation_data_df3[validation_data_df3[column_list[column_list.index(key)]] == 0]["content"]
        label_validation3 = validation_data_df3[validation_data_df3[column_list[column_list.index(key)]] == 0][key]
        if len(label_validation3) == 0:
            continue
        print(key + str(content_validation3.shape))
        print(key + str(label_validation3.shape))
        features_train3 = vectorizer_tfidf.fit_transform(data_dict[key][0])
        features_val3 = vectorizer_tfidf.transform(content_validation3)

        # start format data
        logger.info("start %s data format " % key)
        label_validation3 = np.array(label_validation3)
        save_path = "./predict_samples_classifier_3_val_features_svm_format_files/" + key
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dump_svmlight_file(features_val3, label_validation3, save_path + "/" + key + ".txt",
                           zero_based=True, comment=None, query_id=None)
        logger.info("complete %s data format" % key)
    logger.info("complete all data format")

    #start load model and predict label
    logger.info("start load model and predict label")
    for column in column_list:
        predict_result_save_path3 = "./predict_results_classifier_3/" + column
        if not os.path.exists(predict_result_save_path3):
            os.makedirs(predict_result_save_path3)
        subprocess.call("./thundersvm-master/build/bin/thundersvm-predict " +
                        "./predict_samples_classifier_3_val_features_svm_format_files/" + column + "/" + column + ".txt "
                        + "./model_files_classifier_3/" + column + "/" + column + ".model " +
                        predict_result_save_path3 + "/" + column + "_predict_result.csv", shell=True)
        #add .csv header, to make the next step easier.
        with open(predict_result_save_path3 + "/" + column + "_predict_result.csv", "r", encoding="utf-8") as f:
            temp = []
            for line in f.readlines():
                temp.append(line[0])
            data_frame = pd.DataFrame({column: temp})
            data_frame.to_csv(predict_result_save_path3 + "/" + column + "_predict_result.csv", index=False, sep=",",
                              encoding="utf-8")
        #overwrite the original file,complete add .csv header

        df = pd.read_csv(predict_result_save_path3 + "/" + column + "_predict_result.csv", encoding="utf-8")
        label_new = []
        label = df[column]
        for la in label:
            if la == -1:
                label_new.append(-1)
            else:
                label_new.append(0)

        val_label = validation_data_df3[column]
        count2 = 0
        label_new2 = []
        for li in val_label:
            if li == -2:
                label_new2.append(-2)
            if li == 1:
                label_new2.append(1)
            if li == 0:
                label_new2.append(label_new[count2])
                count2 = count2 + 1
        validation_data_df3[column] = label_new2
    validation_data_df3.to_csv("./predict_results_classifier_3/validation_predict.csv", index=False, sep=",", encoding="utf-8")
    print("complete 20 labels binary classification predict.The result is saved under the path: ./predict_results_classifier_3/validation_predict.csv")
    #stage 3 end
    subprocess.call("python f1_score.py", shell=True)
