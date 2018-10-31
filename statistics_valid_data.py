import pandas as pd
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

df = pd.read_csv("./dataset/data_reform/valid_reform.csv", encoding="utf-8")
f = open("./dataset/statistics_valid_data.txt", "a+")
for column in column_list:
    label_1_percentage = len(df[df[column] == 1][column]) / len(df[column])
    label_0_percentage = len(df[df[column] == 0][column]) / len(df[column])
    label_minus1_percentage = len(df[df[column] == -1][column]) / len(df[column])
    label_minus2_percentage = len(df[df[column] == -2][column]) / len(df[column])
    f.write("the percentage of 1 in " + column + ": " + str(label_1_percentage) + '\n')
    f.write("the percentage of 0 in " + column + ": " + str(label_0_percentage) + '\n')
    f.write("the percentage of -1 in " + column + ": " + str(label_minus1_percentage) + '\n')
    f.write("the percentage of -2 in " + column + ": " + str(label_minus2_percentage) + '\n')
f.close()