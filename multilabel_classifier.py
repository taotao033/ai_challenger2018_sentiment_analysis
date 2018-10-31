#%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
stop_words = []
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
column_drop_list = [
     "id",
     "content",
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

df = pd.read_csv("./dataset/data_reform/train_reform_content_after_cut.csv", encoding="utf-8")
#Number of comments in each category.
df_labels = df.drop(column_drop_list, axis=1)
counts = []
categories = list(df_labels.columns.values)

for i in categories:
    counts.append((i, df_labels[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
#df_stats.to_csv("./output/per_catagory_number_of_contents.csv", encoding='utf-8', index=False, sep=',')

#Using plot tool to show number of comments per category.
df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)
#plt.show()

#How many comments have multiple labels?
rowsums = df.iloc[:, 22:].sum(axis=1)
x = rowsums.value_counts()
#plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)
#plt.show()

"""
print('Percentage of comments that are not labelled:')
print(len(df[(df['location'] == 0) & (df['service'] == 0) & (df['price'] == 0) & (df['environment'] == 0) &
             (df['dish'] == 0) & (df['others'] == 0)]) / len(df))
"""
#output:
"""
Percentage of comments that are not labelled:
0.000419047619047619
"""


"""
#There is no missing comment in comment text column.没有评论为空的
print('Number of missing comments in comment text:')
print(df['content'].isnull().sum())
"""
#output:
"""
Number of missing comments in comment text:
0
"""

#print(df['content'][0])

#Split to train and test sets
#train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)
X_train = df.content
df_val = pd.read_csv("./dataset/data_reform/valid_reform.csv", encoding="utf-8")

val_content = df_val.content
import jieba
content_list = []
for text in val_content:
     token = jieba.cut(text)
     arr_temp = []
     for item in token:
          arr_temp.append(item)
     content_list.append(" ".join(arr_temp))
X_val = content_list
#print(X_train.shape)
#print(X_test.shape)
# LinearSVC
stop_words = ['!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。']
SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])
temp_list = []
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, df[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_val)
    print('Test accuracy is {}'.format(accuracy_score(df_val[category], prediction)))
    temp_list.append(list(prediction))
temp_list_tuple = np.array(temp_list)
location = list(temp_list_tuple[0, ...])
service = list(temp_list_tuple[1, ...])
price = list(temp_list_tuple[2, ...])
environmet = list(temp_list_tuple[3, ...])
dish = list(temp_list_tuple[4, ...])
others = list(temp_list_tuple[5, ...])
prediction_data_fram = pd.DataFrame({'id': df_val['id'], 'content': df_val['content'],
                                     'location': location,
                                     'service': service,
                                     'price': price,
                                     'environment': environmet,
                                     'dish': dish,
                                     'others': others})
prediction_data_fram.to_csv("./output/valid_reform_predict_six_multilabel3.csv", index=False, sep=',')

"""
#output:
... Processing location
Test accuracy is 0.8671284271284271
... Processing service
Test accuracy is 0.8768831168831169
... Processing price
Test accuracy is 0.8707936507936508
... Processing environment
Test accuracy is 0.8806926406926406
... Processing dish
Test accuracy is 0.9713131313131314
... Processing others
Test accuracy is 0.9854834054834055
"""

# Define a pipeline combining a text feature extractor with multi lable classifier
#Naive Bayes
"""
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
#output:
... Processing location
Test accuracy is 0.7397113997113998
... Processing service
Test accuracy is 0.6963347763347764
... Processing price
Test accuracy is 0.7066378066378066
... Processing environment
Test accuracy is 0.6473881673881674
... Processing dish
Test accuracy is 0.9690620490620491
... Processing others
Test accuracy is 0.9855411255411255
"""

"""
#Logistic Regression
LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
temp_list = []
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, df[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_val)
    print('Test accuracy is {}'.format(accuracy_score(df_val[category], prediction)))
    temp_list.append(list(prediction))
temp_list_tuple = np.array(tuple(temp_list))
location = list(temp_list_tuple[0, ...])
service = list(temp_list_tuple[1, ...])
price = list(temp_list_tuple[2, ...])
environmet = list(temp_list_tuple[3, ...])
dish = list(temp_list_tuple[4, ...])
others = list(temp_list_tuple[5, ...])
prediction_data_fram = pd.DataFrame({'id': df_val['id'], 'content': df_val['content'],
                                     'location': location,
                                     'service': service,
                                     'price': price,
                                     'environment': environmet,
                                     'dish': dish,
                                     'others': others})
prediction_data_fram.to_csv("./output/valid_reform_predict_six_multilabel.csv", index=False, sep=',')


... Processing location
Test accuracy is 0.8602308802308802
... Processing service
Test accuracy is 0.8857720057720058
... Processing price
Test accuracy is 0.8803751803751804
... Processing environment
Test accuracy is 0.8842424242424243
... Processing dish
Test accuracy is 0.9704761904761905
... Processing others
Test accuracy is 0.9855411255411255
"""