import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def presion_and_recall(y_predict, y_label, class_type):
    size = len(y_label)

    TP = 0
    FN = 0
    FP = 0

    for i in range(size):

        if y_label[i] == class_type and y_predict[i] == class_type:
            TP += 1
            continue
        if y_label[i] == class_type:
            FN += 1
            continue
        if y_predict[i] == class_type:
            FP += 1

    presion = TP / (TP + FP)
    recall = TP / (TP + FN)

    return presion, recall


test_list = ["location_traffic_convenience"]

key_list1 = [
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
key_list2 = [
    "location",
    "service",
    "price",
    "environment",
    "dish",
    "others"
]


def f1_macro():
    df = pd.read_csv("./predict_results_classifier_3/validation_predict.csv")#predict
    df_val = pd.read_csv("./dataset/valid.csv")#real

    f1_per_column = open("./output/f1_score_thundersvm_c100_g0.5_val.txt", 'a+')
    f1_sum = 0

    for column in key_list1:
        f1 = f1_score(df_val[column], df[column], average='macro')
        f1_per_column.write(column + '-f1_score: ' + str(f1) + '\n')
        f1_sum += f1
    average_f1 = f1_sum / len(key_list1)
    f1_per_column.write('average_f1: ' + str(average_f1))
    f1_per_column.close()
    if average_f1 >= 0.750:
        print("Great! Nice! you win!")
    else:
        print("Unfortunately,come on!")
    print('f1_score: ' + str(average_f1))


f1_macro()
