from keras.models import load_model
import pandas as pd
from models_generator import AttLayer,Attention
from keras.preprocessing.sequence import pad_sequences
import json
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
import jieba
import gc
from keras.utils import multi_gpu_model
#from att_layer import AttentionWeightedAverage
#num_cores = 24
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,
                        #allow_soft_placement=True, device_count={'CPU': num_cores})
#session = tf.Session(config=config)
#K.set_session(session)

#
# config = tf.ConfigProto()
#config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)
#
# KTF.set_session(sess)

list_test = ["location_traffic_convenience"]
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

df = pd.read_csv("./dataset/valid.csv", encoding="utf-8")
content = df["content"]
content_list = []
for text in content:
    token = jieba.cut(text)

    arr_temp = []
    for item in token:
        arr_temp.append(item)
    content_list.append(" ".join(arr_temp))

filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
with open("vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)
    tokenizer.word_index = vocab

content_list_seq = tokenizer.texts_to_sequences(content_list)

content_list_seq_pad = pad_sequences(content_list_seq, maxlen=1000)

for folder in column_list:
    print("processing------------->", folder)
    file_list = os.listdir("./model_files_bigru_attention" + '/' + folder)
    model_file = file_list[0]
    model = load_model("./model_files_bigru_attention/" + folder + "/" + model_file, custom_objects={'Attention': Attention})
    parallel_model = multi_gpu_model(model, gpus=2)
    results = parallel_model.predict(content_list_seq_pad)
    label_list = []
    for item in results:

        idx = np.argmax(item)
        if idx == 0:
            label_list.append(1)
            continue
        if idx == 1:
            label_list.append(0)
            continue
        if idx == 2:
            label_list.append(-1)
            continue
        if idx == 3:
            label_list.append(-2)

    df[folder] = label_list
    del model
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
df.to_csv("./output/predict_validation_20label_bigru.csv", encoding="utf-8", index=False)
