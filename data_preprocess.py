import pandas as pd
import numpy as np
import jieba
import os
from keras.preprocessing.sequence import pad_sequences
import json
from keras.preprocessing.text import Tokenizer

def get_train_data(path, input_length=20):#extract -2,-1,0,1,then start binary classification:define{0(-2), 1(-1,0,1)}
    df = pd.read_csv(path, encoding="utf-8")
    content = df["content"]
    location_traffic_convenience = df["location_traffic_convenience"]
    location_distance_from_business_district = df["location_distance_from_business_district"]
    location_easy_to_find = df["location_easy_to_find"]
    service_wait_time = df["service_wait_time"]
    service_waiters_attitude = df["service_waiters_attitude"]
    service_parking_convenience = df["service_parking_convenience"]
    service_serving_speed = df["service_serving_speed"]
    price_level = df["price_level"]
    price_cost_effective = df["price_cost_effective"]
    price_discount = df["price_discount"]
    environment_decoration = df["environment_decoration"]
    environment_noise = df["environment_noise"]
    environment_space = df["environment_space"]
    environment_cleaness = df["environment_cleaness"]
    dish_portion = df["dish_portion"]
    dish_taste = df["dish_taste"]
    dish_look = df["dish_look"]
    dish_recommendation = df["dish_recommendation"]
    others_overall_experience = df["others_overall_experience"]
    others_willing_to_consume_again = df["others_willing_to_consume_again"]

    label_list = []
    for i, review in enumerate(content):
        label = [
            location_traffic_convenience[i],
            location_distance_from_business_district[i],
            location_easy_to_find[i],
            service_wait_time[i],
            service_waiters_attitude[i],
            service_parking_convenience[i],
            service_serving_speed[i],
            price_level[i],
            price_cost_effective[i],
            price_discount[i],
            environment_decoration[i],
            environment_noise[i],
            environment_space[i],
            environment_cleaness[i],
            dish_portion[i],
            dish_taste[i],
            dish_look[i],
            dish_recommendation[i],
            others_overall_experience[i],
            others_willing_to_consume_again[i]
        ]
        for idx in range(len(label)):
            if label[idx] == -2:
                label[idx] = 0
            else:
                label[idx] = 1

        label_list.append(label)

    label_list = np.array(label_list)
    """
    content = df['content']

    content_list = []
    for text in content:
        token = jieba.cut(text)

        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_list.append(" ".join(arr_temp))
    
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")

    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)#fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content)# 返回值：序列的列表，列表中每个序列对应于一段输入文本(即：将文本转化为标量序列)
    content_list_seq_pad = pad_sequences(content_list_seq, padding='pre', maxlen=input_length)#将长为 nb_samples 的序列（标量序列）转化为形如 (nb_samples,nb_timesteps) 2D numpy array。
                                                                                              # 如果提供了参数 maxlen ， nb_timesteps=maxlen ，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。
                                                                                             #padding： ‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

    return content_list_seq_pad, label_list, len(vocab) + 1, 2
    """
    return label_list

def get_train_data_2(path,):#only extract -1,0,1,then start binary classification:define{0(-1,0), 1(1)}
    df = pd.read_csv(path, encoding="utf-8")

    #content = df["content"]
    location_traffic_convenience = df[df["location_traffic_convenience"] != -2]
    location_distance_from_business_district = df[df["location_distance_from_business_district"] != -2]
    location_easy_to_find = df[df["location_easy_to_find"] != -2]
    service_wait_time = df[df["service_wait_time"] != -2]
    service_waiters_attitude = df[df["service_waiters_attitude"] != -2]
    service_parking_convenience = df[df["service_parking_convenience"] != -2]
    service_serving_speed = df[df["service_serving_speed"] != -2]
    price_level = df[df["price_level"] != -2]
    price_cost_effective = df[df["price_cost_effective"] != -2]
    price_discount = df[df["price_discount"] != -2]
    environment_decoration = df[df["environment_decoration"] != -2]
    environment_noise = df[df["environment_noise"] != -2]
    environment_space = df[df["environment_space"] != -2]
    environment_cleaness = df[df["environment_cleaness"] != -2]
    dish_portion = df[df["dish_portion"] != -2]
    dish_taste = df[df["dish_taste"] != -2]
    dish_look = df[df["dish_look"] != -2]
    dish_recommendation = df[df["dish_recommendation"] != -2]
    others_overall_experience = df[df["others_overall_experience"] != -2]
    others_willing_to_consume_again = df[df["others_willing_to_consume_again"] != -2]

    location_traffic_convenience_content = location_traffic_convenience["content"]
    location_distance_from_business_district_content = location_distance_from_business_district["content"]
    location_easy_to_find_content = location_easy_to_find["content"]
    service_wait_time_content = service_wait_time["content"]
    service_waiters_attitude_content = service_waiters_attitude["content"]
    service_parking_convenience_content = service_parking_convenience["content"]
    service_serving_speed_content = service_serving_speed["content"]
    price_level_content = price_level["content"]
    price_cost_effective_content = price_cost_effective["content"]
    price_discount_content = price_discount["content"]
    environment_decoration_content = environment_decoration["content"]
    environment_noise_content = environment_noise["content"]
    environment_space_content = environment_space["content"]
    environment_cleaness_content = environment_cleaness["content"]
    dish_portion_content = dish_portion["content"]
    dish_taste_content = dish_taste["content"]
    dish_look_content = dish_look["content"]
    dish_recommendation_content = dish_recommendation["content"]
    others_overall_experience_content = others_overall_experience["content"]
    others_willing_to_consume_again_content = others_willing_to_consume_again["content"]

    location_traffic_convenience_label = location_traffic_convenience["location_traffic_convenience"]
    location_distance_from_business_district_label = location_distance_from_business_district["location_distance_from_business_district"]
    location_easy_to_find_label = location_easy_to_find["location_easy_to_find"]
    service_wait_time_label = service_wait_time["service_wait_time"]
    service_waiters_attitude_label = service_waiters_attitude["service_waiters_attitude"]
    service_parking_convenience_label = service_parking_convenience["service_parking_convenience"]
    service_serving_speed_label = service_serving_speed["service_serving_speed"]
    price_level_label = price_level["price_level"]
    price_cost_effective_label = price_cost_effective["price_cost_effective"]
    price_discount_label = price_discount["price_discount"]
    environment_decoration_label = environment_decoration["environment_decoration"]
    environment_noise_label = environment_noise["environment_noise"]
    environment_space_label = environment_space["environment_space"]
    environment_cleaness_label = environment_cleaness["environment_cleaness"]
    dish_portion_label = dish_portion["dish_portion"]
    dish_taste_label = dish_taste["dish_taste"]
    dish_look_label = dish_look["dish_look"]
    dish_recommendation_label = dish_recommendation["dish_recommendation"]
    others_overall_experience_label = others_overall_experience["others_overall_experience"]
    others_willing_to_consume_again_label = others_willing_to_consume_again["others_willing_to_consume_again"]
    data_dic = {
        "location_traffic_convenience": [location_traffic_convenience_content, location_traffic_convenience_label],

        "location_distance_from_business_district": [location_distance_from_business_district_content,
                                                     location_distance_from_business_district_label],
        "location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
        "service_wait_time": [service_wait_time_content, service_wait_time_label],
        "service_waiters_attitude": [service_waiters_attitude_content, service_waiters_attitude_label],
        "service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
        "service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
        "price_level": [price_level_content, price_level_label],
        "price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
        "price_discount": [price_discount_content, price_discount_label],
        "environment_decoration": [environment_decoration_content, environment_decoration_label],
        "environment_noise": [environment_noise_content, environment_noise_label],
        "environment_space": [environment_space_content, environment_space_label],
        "environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
        "dish_portion": [dish_portion_content, dish_portion_label],
        "dish_taste": [dish_taste_content, dish_taste_label],
        "dish_look": [dish_look_content, dish_look_label],
        "dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
        "others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
        "others_willing_to_consume_again": [others_willing_to_consume_again_content,
                                            others_willing_to_consume_again_label]

    }

    for key in data_dic.keys():

        label_list = data_dic[key][1]

        label_list_new = []
        for data in label_list:

            if data == 1:
                label_list_new.append(1)
            else:
                label_list_new.append(0)
        data_dic[key][1] = label_list_new

    return data_dic


def get_train_data_3(path,):#only extract 0,-1,then start binary classification{0, -1}
    df = pd.read_csv(path, encoding="utf-8")

    #content = df["content"]
    location_traffic_convenience = df[(df["location_traffic_convenience"] != -2) &
                                      (df["location_traffic_convenience"] != 1)]
    location_distance_from_business_district = df[(df["location_distance_from_business_district"] != -2) &
                                                  (df["location_distance_from_business_district"] != 1)]
    location_easy_to_find = df[(df["location_easy_to_find"] != -2) &
                               (df["location_easy_to_find"] != 1)]
    service_wait_time = df[(df["service_wait_time"] != -2) &
                           (df["service_wait_time"] != 1)]
    service_waiters_attitude = df[(df["service_waiters_attitude"] != -2) &
                                  (df["service_waiters_attitude"] != 1)]
    service_parking_convenience = df[(df["service_parking_convenience"] != -2) &
                                     (df["service_parking_convenience"] != 1)]
    service_serving_speed = df[(df["service_serving_speed"] != -2) & (df["service_serving_speed"] != 1)]
    price_level = df[(df["price_level"] != -2) & (df["price_level"] != 1)]
    price_cost_effective = df[(df["price_cost_effective"] != -2) & (df["price_cost_effective"] != 1)]
    price_discount = df[(df["price_discount"] != -2) & (df["price_discount"] != 1)]
    environment_decoration = df[(df["environment_decoration"] != -2) & (df["environment_decoration"] != 1)]
    environment_noise = df[(df["environment_noise"] != -2) & (df["environment_noise"] != 1)]
    environment_space = df[(df["environment_space"] != -2) & (df["environment_space"] != 1)]
    environment_cleaness = df[(df["environment_cleaness"] != -2) & (df["environment_cleaness"] != 1)]
    dish_portion = df[(df["dish_portion"] != -2) & (df["dish_portion"] != 1)]
    dish_taste = df[(df["dish_taste"] != -2) & (df["dish_taste"] != 1)]
    dish_look = df[(df["dish_look"] != -2) & (df["dish_look"] != 1)]
    dish_recommendation = df[(df["dish_recommendation"] != -2) & (df["dish_recommendation"] != 1)]
    others_overall_experience = df[(df["others_overall_experience"] != -2) &
                                   (df["others_overall_experience"] != 1)]
    others_willing_to_consume_again = df[(df["others_willing_to_consume_again"] != -2) &
                                         (df["others_willing_to_consume_again"] != 1)]

    location_traffic_convenience_content = location_traffic_convenience["content"]
    location_distance_from_business_district_content = location_distance_from_business_district["content"]
    location_easy_to_find_content = location_easy_to_find["content"]
    service_wait_time_content = service_wait_time["content"]
    service_waiters_attitude_content = service_waiters_attitude["content"]
    service_parking_convenience_content = service_parking_convenience["content"]
    service_serving_speed_content = service_serving_speed["content"]
    price_level_content = price_level["content"]
    price_cost_effective_content = price_cost_effective["content"]
    price_discount_content = price_discount["content"]
    environment_decoration_content = environment_decoration["content"]
    environment_noise_content = environment_noise["content"]
    environment_space_content = environment_space["content"]
    environment_cleaness_content = environment_cleaness["content"]
    dish_portion_content = dish_portion["content"]
    dish_taste_content = dish_taste["content"]
    dish_look_content = dish_look["content"]
    dish_recommendation_content = dish_recommendation["content"]
    others_overall_experience_content = others_overall_experience["content"]
    others_willing_to_consume_again_content = others_willing_to_consume_again["content"]

    location_traffic_convenience_label = location_traffic_convenience["location_traffic_convenience"]
    location_distance_from_business_district_label = location_distance_from_business_district["location_distance_from_business_district"]
    location_easy_to_find_label = location_easy_to_find["location_easy_to_find"]
    service_wait_time_label = service_wait_time["service_wait_time"]
    service_waiters_attitude_label = service_waiters_attitude["service_waiters_attitude"]
    service_parking_convenience_label = service_parking_convenience["service_parking_convenience"]
    service_serving_speed_label = service_serving_speed["service_serving_speed"]
    price_level_label = price_level["price_level"]
    price_cost_effective_label = price_cost_effective["price_cost_effective"]
    price_discount_label = price_discount["price_discount"]
    environment_decoration_label = environment_decoration["environment_decoration"]
    environment_noise_label = environment_noise["environment_noise"]
    environment_space_label = environment_space["environment_space"]
    environment_cleaness_label = environment_cleaness["environment_cleaness"]
    dish_portion_label = dish_portion["dish_portion"]
    dish_taste_label = dish_taste["dish_taste"]
    dish_look_label = dish_look["dish_look"]
    dish_recommendation_label = dish_recommendation["dish_recommendation"]
    others_overall_experience_label = others_overall_experience["others_overall_experience"]
    others_willing_to_consume_again_label = others_willing_to_consume_again["others_willing_to_consume_again"]
    data_dic = {
        "location_traffic_convenience": [location_traffic_convenience_content, location_traffic_convenience_label],

        "location_distance_from_business_district": [location_distance_from_business_district_content,
                                                     location_distance_from_business_district_label],
        "location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
        "service_wait_time": [service_wait_time_content, service_wait_time_label],
        "service_waiters_attitude": [service_waiters_attitude_content, service_waiters_attitude_label],
        "service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
        "service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
        "price_level": [price_level_content, price_level_label],
        "price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
        "price_discount": [price_discount_content, price_discount_label],
        "environment_decoration": [environment_decoration_content, environment_decoration_label],
        "environment_noise": [environment_noise_content, environment_noise_label],
        "environment_space": [environment_space_content, environment_space_label],
        "environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
        "dish_portion": [dish_portion_content, dish_portion_label],
        "dish_taste": [dish_taste_content, dish_taste_label],
        "dish_look": [dish_look_content, dish_look_label],
        "dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
        "others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
        "others_willing_to_consume_again": [others_willing_to_consume_again_content,
                                            others_willing_to_consume_again_label]

    }

    # for key in data_dic.keys():
    #
    #     label_list = data_dic[key][1]
    #
    #     label_list_new = []
    #     for data in label_list:
    #
    #         if data == 1:
    #             label_list_new.append(1)
    #         else:
    #             label_list_new.append(0)
    #     data_dic[key][1] = label_list_new

    return data_dic


def get_val_data(path, input_length=20):
    df = pd.read_csv(path, encoding="utf-8")
    content = df["content"]
    location_traffic_convenience = df["location_traffic_convenience"]
    location_distance_from_business_district = df["location_distance_from_business_district"]
    location_easy_to_find = df["location_easy_to_find"]
    service_wait_time = df["service_wait_time"]
    service_waiters_attitude = df["service_waiters_attitude"]
    service_parking_convenience = df["service_parking_convenience"]
    service_serving_speed = df["service_serving_speed"]
    price_level = df["price_level"]
    price_cost_effective = df["price_cost_effective"]
    price_discount = df["price_discount"]
    environment_decoration = df["environment_decoration"]
    environment_noise = df["environment_noise"]
    environment_space = df["environment_space"]
    environment_cleaness = df["environment_cleaness"]
    dish_portion = df["dish_portion"]
    dish_taste = df["dish_taste"]
    dish_look = df["dish_look"]
    dish_recommendation = df["dish_recommendation"]
    others_overall_experience = df["others_overall_experience"]
    others_willing_to_consume_again = df["others_willing_to_consume_again"]

    label_list = []
    for i, review in enumerate(content):
        label = [
            location_traffic_convenience[i],
            location_distance_from_business_district[i],
            location_easy_to_find[i],
            service_wait_time[i],
            service_waiters_attitude[i],
            service_parking_convenience[i],
            service_serving_speed[i],
            price_level[i],
            price_cost_effective[i],
            price_discount[i],
            environment_decoration[i],
            environment_noise[i],
            environment_space[i],
            environment_cleaness[i],
            dish_portion[i],
            dish_taste[i],
            dish_look[i],
            dish_recommendation[i],
            others_overall_experience[i],
            others_willing_to_consume_again[i]
        ]
        for idx in range(len(label)):
            if label[idx] == -2:
                label[idx] = 0
            else:
                label[idx] = 1

        label_list.append(label)

    label_list = np.array(label_list)
    """
    content = df['content']

    content_list = []
    for text in content:
        token = jieba.cut(text)

        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_list.append(" ".join(arr_temp))
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")

    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content_list)  # fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content_list)  # 返回值：序列的列表，列表中每个序列对应于一段输入文本(即：将文本转化为标量序列)
    content_list_seq_pad = pad_sequences(content_list_seq, padding='pre',
                                         maxlen=input_length)  # 将长为 nb_samples 的序列（标量序列）转化为形如 (nb_samples,nb_timesteps) 2D numpy array。
    # 如果提供了参数 maxlen ， nb_timesteps=maxlen ，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。
    # padding： ‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

    return content_list_seq_pad, label_list, len(vocab) + 1, 2
    """
    return label_list


def get_boject_six_matrix(input_length, path):

    content_list_seq_pad, label_list, len_vocab, len_label = get_train_data(input_length, path)

    location_label = label_list[..., 0:3]
    service_label = label_list[..., 3:7]
    price_label = label_list[..., 7:10]
    environment_label = label_list[..., 10:14]
    dish_label = label_list[..., 14:18]
    others_label = label_list[..., 18:20]

    location_label_list = []
    for item in location_label:
        x = np.max(item)
        if x == 0:
            location_label_list.append(0)
            continue
        if x == 1:
            location_label_list.append(1)

    #print(location_label_list)

    service_label_list = []
    for item in service_label:
        x = np.max(item)
        if x == 0:
            service_label_list.append(0)
            continue
        if x == 1:
            service_label_list.append(1)

    #print(service_label_list)

    price_label_list = []
    for item in price_label:
        x = np.max(item)
        if x == 0:
            price_label_list.append(0)
            continue
        if x == 1:
            price_label_list.append(1)

    #print(price_label_list)

    environment_label_list = []
    for item in environment_label:
        x = np.max(item)
        if x == 0:
            environment_label_list.append(0)
            continue
        if x == 1:
            environment_label_list.append(1)

    #print(environment_label_list)

    dish_label_list = []
    for item in dish_label:
        x = np.max(item)
        if x == 0:
            dish_label_list.append(0)
            continue
        if x == 1:
            dish_label_list.append(1)

    #print(dish_label_list)

    others_label_list = []
    for item in others_label:
        x =np.max(item)
        if x == 0:
            others_label_list.append(0)
            continue
        if x == 1:
            others_label_list.append(1)

    #print(others_label_list)

    array_empty = np.empty([len(location_label_list), 6], dtype=int)
    array_empty[..., 0] = np.array(location_label_list)
    array_empty[..., 1] = np.array(service_label_list)
    array_empty[..., 2] = np.array(price_label_list)
    array_empty[..., 3] = np.array(environment_label_list)
    array_empty[..., 4] = np.array(dish_label_list)
    array_empty[..., 5] = np.array(others_label_list)

    object_six_matrix_label = array_empty
    object_six_matrix_column_num = 6
    return content_list_seq_pad, object_six_matrix_label, len_vocab, len_label, object_six_matrix_column_num


def get_segment_train_data(input_length, path):#extract the data that made up of[-2 , -1, 0, 1] ,and -2 label is a very small percentage of them
    df = pd.read_csv(path, encoding="utf-8")

    content = df["content"]
    """
    content_list = []
    for text in content:
        token = jieba.cut(text)
        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_list.append(" ".join(arr_temp))
    """
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)
     
    content_list_seq = tokenizer.texts_to_sequences(content)
    # print(sum([len(c) for c in content_list_seq])/len(content_list_seq))
    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)
    temp = []
    for content_l in content_list_seq_pad:
        temp.append(content_l)

    # print(content_list_seq_pad[0])
    df["content"] = temp

    location = df[df["location"] != 0]
    service = df[df["service"] != 0]
    price = df[df["price"] != 0]
    environment = df[df["environment"] != 0]
    dish = df[df["dish"] != 0]
    others = df[df["others"] != 0]

    location_reform = location[(location["location_traffic_convenience"] != 1) &
                               (location["location_traffic_convenience"] != -2)]

    service_reform = service[service["service_waiters_attitude"] != 1]

    location_traffic_convenience_label = location_reform["location_traffic_convenience"]
    location_distance_from_business_district_label = location["location_distance_from_business_district"]
    location_easy_to_find_label = location["location_easy_to_find"]
    service_wait_time_label = service["service_wait_time"]
    service_waiters_attitude_label = service_reform["service_waiters_attitude"]
    service_parking_convenience_label = service["service_parking_convenience"]
    service_serving_speed_label = service["service_serving_speed"]
    price_level_label = price["price_level"]
    price_cost_effective_label = price["price_cost_effective"]
    price_discount_label = price["price_discount"]
    environment_decoration_label = environment["environment_decoration"]
    environment_noise_label = environment["environment_noise"]
    environment_space_label = environment["environment_space"]
    environment_cleaness_label = environment["environment_cleaness"]
    dish_portion_label = dish["dish_portion"]
    dish_taste_label = dish["dish_taste"]
    dish_look_label = dish["dish_look"]
    dish_recommendation_label = dish["dish_recommendation"]
    others_overall_experience_label = others["others_overall_experience"]
    others_willing_to_consume_again_label = others["others_willing_to_consume_again"]

    location_traffic_convenience_content = location_reform["content"]
    location_distance_from_business_district_content = location["content"]
    location_easy_to_find_content = location["content"]
    service_wait_time_content = service["content"]
    service_waiters_attitude_content = service_reform["content"]
    service_parking_convenience_content = service["content"]
    service_serving_speed_content = service["content"]
    price_level_content = price["content"]
    price_cost_effective_content = price["content"]
    price_discount_content = price["content"]
    environment_decoration_content = environment["content"]
    environment_noise_content = environment["content"]
    environment_space_content = environment["content"]
    environment_cleaness_content = environment["content"]
    dish_portion_content = dish["content"]
    dish_taste_content = dish["content"]
    dish_look_content = dish["content"]
    dish_recommendation_content = dish["content"]
    others_overall_experience_content = others["content"]
    others_willing_to_consume_again_content = others["content"]

    location_reform2 = location[location["location_traffic_convenience"] == 1]
    location_traffic_convenience_label_add_only_label1 = location_reform2["location_traffic_convenience"]
    location_traffic_convenience_content_add_add_only_label1 = location_reform2["content"]

    location_reform2_lo = location[location["location_traffic_convenience"] == -2]
    location_traffic_convenience_label_add_only_label_minus2 = location_reform2_lo["location_traffic_convenience"]
    location_traffic_convenience_content_add_add_only_label_minus2 = location_reform2_lo["content"]

    frames_label1 = [location_traffic_convenience_label, location_traffic_convenience_label_add_only_label1[0:20000],
                     location_traffic_convenience_label_add_only_label_minus2]
    location_traffic_convenience_label_concat = pd.concat(frames_label1)

    frames_content1 = [location_traffic_convenience_content, location_traffic_convenience_content_add_add_only_label1[0:20000],
                       location_traffic_convenience_content_add_add_only_label_minus2]
    location_traffic_convenience_content_concat = pd.concat(frames_content1)


    service_reform2 = service[service["service_waiters_attitude"] == 1]
    service_waiters_attitude_label_add_only_label1 = service_reform2["service_waiters_attitude"]
    service_waiters_attitude_content_add_only_label1 = service_reform2["content"]
    frames_label2 = [service_waiters_attitude_label, service_waiters_attitude_label_add_only_label1[0:21372]]
    service_waiters_attitude_label_concat = pd.concat(frames_label2)
    frames_content2 = [service_waiters_attitude_content, service_waiters_attitude_content_add_only_label1[0:21372]]
    service_waiters_attitude_content_concat = pd.concat(frames_content2)


    segment_data_dic = {
        "location_traffic_convenience": [location_traffic_convenience_content_concat, location_traffic_convenience_label_concat],

        "location_distance_from_business_district": [location_distance_from_business_district_content,
                                                     location_distance_from_business_district_label],
        "location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
        "service_wait_time": [service_wait_time_content, service_wait_time_label],
        "service_waiters_attitude": [service_waiters_attitude_content_concat, service_waiters_attitude_label_concat],
        "service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
        "service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
        "price_level": [price_level_content, price_level_label],
        "price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
        "price_discount": [price_discount_content, price_discount_label],
        "environment_decoration": [environment_decoration_content, environment_decoration_label],
        "environment_noise": [environment_noise_content, environment_noise_label],
        "environment_space": [environment_space_content, environment_space_label],
        "environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
        "dish_portion": [dish_portion_content, dish_portion_label],
        "dish_taste": [dish_taste_content, dish_taste_label],
        "dish_look": [dish_look_content, dish_look_label],
        "dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
        "others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
        "others_willing_to_consume_again": [others_willing_to_consume_again_content,
                                            others_willing_to_consume_again_label]

    }
    for key in segment_data_dic.keys():

        label_list = segment_data_dic[key][1]

        label_list_onehot = []
        for data in label_list:

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
        segment_data_dic[key][1] = label_list_onehot

    #return segment_data_dic, len(vocab), 4
    return segment_data_dic,  4


def get_segment_train_data2(input_length, path):#extract the data that made up of -2 in whole.
    df = pd.read_csv(path, encoding="utf-8")

    content = df["content"]
    """
    content_list = []
    for text in content:
        token = jieba.cut(text)
        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_list.append(" ".join(arr_temp))
     """
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content)
    # print(sum([len(c) for c in content_list_seq])/len(content_list_seq))
    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)
    temp = []
    for content_l in content_list_seq_pad:
        temp.append(content_l)

    # print(content_list_seq_pad[0])
    df["content"] = temp

    location = df[df["location"] != 1]
    service = df[df["service"] != 1]
    price = df[df["price"] != 1]
    environment = df[df["environment"] != 1]
    dish = df[df["dish"] != 1]
    others = df[df["others"] != 1]

    location_traffic_convenience_label = location["location_traffic_convenience"]
    location_distance_from_business_district_label = location["location_distance_from_business_district"]
    location_easy_to_find_label = location["location_easy_to_find"]
    service_wait_time_label = service["service_wait_time"]
    service_waiters_attitude_label = service["service_waiters_attitude"]
    service_parking_convenience_label = service["service_parking_convenience"]
    service_serving_speed_label = service["service_serving_speed"]
    price_level_label = price["price_level"]
    price_cost_effective_label = price["price_cost_effective"]
    price_discount_label = price["price_discount"]
    environment_decoration_label = environment["environment_decoration"]
    environment_noise_label = environment["environment_noise"]
    environment_space_label = environment["environment_space"]
    environment_cleaness_label = environment["environment_cleaness"]
    dish_portion_label = dish["dish_portion"]
    dish_taste_label = dish["dish_taste"]
    dish_look_label = dish["dish_look"]
    dish_recommendation_label = dish["dish_recommendation"]
    others_overall_experience_label = others["others_overall_experience"]
    others_willing_to_consume_again_label = others["others_willing_to_consume_again"]

    location_traffic_convenience_content = location["content"]
    location_distance_from_business_district_content = location["content"]
    location_easy_to_find_content = location["content"]
    service_wait_time_content = service["content"]
    service_waiters_attitude_content = service["content"]
    service_parking_convenience_content = service["content"]
    service_serving_speed_content = service["content"]
    price_level_content = price["content"]
    price_cost_effective_content = price["content"]
    price_discount_content = price["content"]
    environment_decoration_content = environment["content"]
    environment_noise_content = environment["content"]
    environment_space_content = environment["content"]
    environment_cleaness_content = environment["content"]
    dish_portion_content = dish["content"]
    dish_taste_content = dish["content"]
    dish_look_content = dish["content"]
    dish_recommendation_content = dish["content"]
    others_overall_experience_content = others["content"]
    others_willing_to_consume_again_content = others["content"]
    segment_data_dic = {
        "location_traffic_convenience": [location_traffic_convenience_content, location_traffic_convenience_label],

        "location_distance_from_business_district": [location_distance_from_business_district_content,
                                                     location_distance_from_business_district_label],
        "location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
        "service_wait_time": [service_wait_time_content, service_wait_time_label],
        "service_waiters_attitude": [service_waiters_attitude_content, service_waiters_attitude_label],
        "service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
        "service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
        "price_level": [price_level_content, price_level_label],
        "price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
        "price_discount": [price_discount_content, price_discount_label],
        "environment_decoration": [environment_decoration_content, environment_decoration_label],
        "environment_noise": [environment_noise_content, environment_noise_label],
        "environment_space": [environment_space_content, environment_space_label],
        "environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
        "dish_portion": [dish_portion_content, dish_portion_label],
        "dish_taste": [dish_taste_content, dish_taste_label],
        "dish_look": [dish_look_content, dish_look_label],
        "dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
        "others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
        "others_willing_to_consume_again": [others_willing_to_consume_again_content,
                                            others_willing_to_consume_again_label]

    }
    for key in segment_data_dic.keys():

        label_list = segment_data_dic[key][1]

        label_list_onehot = []
        for data in label_list:

            label = [0, 0, 0, 0]

            #if data == 1:
             #   label[0] = 1
            #if data == 0:
             #   label[1] = 1
            #if data == -1:
             #   label[2] = 1
            if data == -2:
                label[3] = 1
            label_list_onehot.append(label)
        segment_data_dic[key][1] = label_list_onehot

    # return segment_data_dic, len(vocab), 4
    return segment_data_dic, 4

def get_multilabel_train_data(input_length, path):
    df = pd.read_csv(path, encoding="utf-8")
    content = df["content"]
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content)
    # print(sum([len(c) for c in content_list_seq])/len(content_list_seq))
    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)

    return df, content_list_seq_pad, 4, len(vocab)

if __name__ == '__main__':
    length = 300
    content_list_seq_pad, object_six_matrix_label, len_vocab, len_label, object_six_matrix_column_num = get_boject_six_matrix(input_length=length, path="./dataset/valid.csv")
    data = pd.read_csv("./dataset/valid.csv", encoding='utf-8')
    #my_matrix = np.loadtxt(open("./dataset/train_mini.csv", "rb"), delimiter=",", skiprows=2)
    #data2 = pd.read_excel("./dataset/train_data_after_cut.xlsx", encodeing='utf-8')

    label_twenty = data[['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find',
                        'service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed',
                        'price_level','price_cost_effective','price_discount',
                        'environment_decoration','environment_noise','environment_space','environment_cleaness',
                        'dish_portion','dish_taste','dish_look','dish_recommendation',
                        'others_overall_experience','others_willing_to_consume_again']].values
    id = list(data['id'])
    content = list(data['content'])
    location_traffic_convenience = list(data['location_traffic_convenience'])
    location_distance_from_business_district = list(data['location_distance_from_business_district'])
    location_easy_to_find = list(data['location_easy_to_find'])
    service_wait_time = list(data['service_wait_time'])
    service_waiters_attitude = list(data['service_waiters_attitude'])
    service_parking_convenience = list(data['service_parking_convenience'])
    service_serving_speed = list(data['service_serving_speed'])
    price_level = list(data['price_level'])
    price_cost_effective = list(data['price_cost_effective'])
    price_discount = list(data['price_discount'])
    environment_decoration = list(data['environment_decoration'])
    environment_noise = list(data['environment_noise'])
    environment_space = list(data['environment_space'])
    environment_cleaness = list(data['environment_cleaness'])
    dish_portion = list(data['dish_portion'])
    dish_taste = list(data['dish_taste'])
    dish_look = list(data['dish_look'])
    dish_recommendation = list(data['dish_recommendation'])
    others_overall_experience = list(data['others_overall_experience'])
    others_willing_to_consume_again = list(data['others_willing_to_consume_again'])
    location = list(object_six_matrix_label[..., 0])
    service = list(object_six_matrix_label[..., 1])
    price = list(object_six_matrix_label[..., 2])
    environment = list(object_six_matrix_label[..., 3])
    dish = list(object_six_matrix_label[..., 4])
    others = list(object_six_matrix_label[..., 5])

    dataframe =  pd.DataFrame({'id':id,'content':content,
                               'location_traffic_convenience':location_traffic_convenience,
                               'location_distance_from_business_district':location_distance_from_business_district,
                               'location_easy_to_find':location_easy_to_find,
                               'service_wait_time':service_wait_time,
                               'service_waiters_attitude':service_waiters_attitude,
                               'service_parking_convenience':service_parking_convenience,
                               'service_serving_speed':service_serving_speed,
                               'price_level':price_level,
                               'price_cost_effective':price_cost_effective,
                               'price_discount':price_discount,
                               'environment_decoration':environment_decoration,
                               'environment_noise':environment_noise,
                               'environment_space':environment_space,
                               'environment_cleaness':environment_cleaness,
                               'dish_portion':dish_portion,
                               'dish_taste':dish_taste,
                               'dish_look':dish_look,
                               'dish_recommendation':dish_recommendation,
                               'others_overall_experience':others_overall_experience,
                               'others_willing_to_consume_again':others_willing_to_consume_again,
                               'location':location,
                               'service':service,
                               'price':price,
                               'environment':environment,
                               'dish':dish,
                               'others':others})
    #print(type(content))
    dataframe.to_csv("./dataset/data_reform/valid_reform.csv", index=False, sep=',')
    #print(list(object_six_matrix_label[..., 0]))
    #print('\n')
    #print(label_twenty[..., 0])
    #print(content)
    #print(dataframe)
