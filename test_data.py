import pandas as pd
import numpy as np
from data_preprocess import get_train_data_2
#df = pd.read_csv("./dataset/data_reform/valid_reform.csv", encoding="utf-8")

#print(len(df["location_easy_to_find"] == -2))
#data_output = df[df["location_easy_to_find"] == 1]
#data_output.to_csv("./test.csv", index=False, sep=',')
#print(data_output["location_easy_to_find"])
#print(df[df["location_traffic_convenience"] != 1])
#print(len(df["location_traffic_convenience"] != -2))
#print(df[df["location_traffic_convenience"] != -2])
#print(len((df["location_traffic_convenience"] != 1) & (df["location_traffic_convenience"] != -2)))
#print(pd.concat([df["location_distance_from_business_district"], df["location_traffic_convenience"][0:100]]))
#location_reform = location[(location["location_traffic_convenience"] != 1) &
#                               (location["location_traffic_convenience"] != -2)]
# f_in = open("svm_output_val.txt", "r")
# f_out = open("svm_output_val_mini.txt", "a+")
# count = 1000
# for line in f_in.readlines():
#     count = count - 1
#     #line = line.replace("\n", "")
#     f_out.write(line)
#     if count == 0:
#         break
# f_in.close()
# f_out.close()
# df = pd.read_csv("./output/svm_output_val_mini_predict.csv", encoding='utf-8')
# print(df["location_traffic_convenience"])

# f_in = open("./svm_output_val_mini.txt", "r")
# #f_out = open("./output/svm_output_val_mini_real.txt", "w")
# list = []
# for line in f_in.readlines():
#     column = line.split()
#     #rint(column)
#     label = column[0]
#     print(label)
#     #print(type(label))
#     #f_out.write(label)
#     list.append(label)
# # f_in.close()
# # f_out.close()
# list_new = np.array(list)
# data_frame = pd.DataFrame({"location_traffic_convenience": list})
# data_frame.to_csv("./output/svm_output_val_mini_real.csv", index=False, sep=',', encoding='utf-8')

# data_dict = get_train_data_2(path="./dataset/train_mini.csv",)
# for key in data_dict.keys():
#     print(np.array(data_dict[key][1]).ndim)

#df = pd.read_csv("./dataset/valid_mini.csv", encoding="utf-8")
# location_easy_to_find = df[(df["location_easy_to_find"] != -2) & (df["location_easy_to_find"] != 1)]
# print(location_easy_to_find["location_easy_to_find"])
# print(len(location_easy_to_find["location_easy_to_find"]))

# from data_preprocess import get_train_data_3
# data_dict = get_train_data_3("./dataset/valid_mini.csv")
# for key in data_dict:
#     print(data_dict[key][1])

# f_in = open("./svm_dataset/svm_output_val.txt", "r", encoding="utf-8")
# f_out = open("./svm_dataset/svm_out_val_mini_10000lines.txt", "a", encoding="utf-8")
# count = 0
# for line in f_in.readlines():
#     f_out.write(line)
#     count = count + 1
#     if count == 10000:
#         break

#df = pd.read_csv("./svm_dataset/svm_out_val_mini_5000lines_predict.csv", encoding="utf-8")

#
#df.to_csv("aaa.csv", index=False, sep=",", encoding="utf-8")
import csv
# with open("./svm_dataset/svm_out_val_mini_5000lines_predict.csv", "r") as csvfile:
#     list = []
#     for row in csvfile.readlines():
#          list.append(row[0])
# data_frame = pd.DataFrame({"label": list})
# data_frame.to_csv("yes.csv", index=False, sep=",", encoding="utf-8")
# df1 = pd.read_csv("1.csv", encoding="utf-8")
# #df2 = pd.read_csv("2.csv", encoding="utf-8")
#
# data = df1[df1["dish_look"] == 1]["id"]
# data2 = df1[df1["dish_look"] == 0]["id"]
# frame = [data, data2]
# print(data)
# print('\n')
# print(data2)
# print('\n')
# print(pd.concat(frame))
# data3 = pd.concat(frame)
#
# validation_data_df3 = pd.read_csv("./predict_results_classifier_2/validation_predict.csv", encoding="utf-8")
# len = len(validation_data_df3["location_distance_from_business_district"])
#
# print(len)

str = " 什么 ， 看到 某 团购"