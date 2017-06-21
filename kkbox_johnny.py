import pandas as pd
import pyprind
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np

df = pd.read_csv('data/kkbox/events_train.csv')
df_label = pd.read_csv('data/kkbox/labels_train.csv')

data_list = df.values[:, :]
label_list = df_label.values[:, :]


title_id = []
user_id = []

print("---- csv 讀取結束 ----")
print("---- train data 計算數量 開始 ----")

for item in data_list:
    t_id = item[2]
    u_id = item[1]

    title_id.append(t_id)
    user_id.append(u_id)

user_set = set(user_id)
title_set = set(title_id)
user_size = len(user_set)
title_size = len(title_set)

print("user count {}".format(user_size))
print("title count {}".format(title_size))
print("---- train data 計算數量 完畢 ----")

print("---- train label 計算數量 開始 ----")
label_user = []
label_title = []
for item in label_list:
    u_id = int(item[0])
    t_id = int(item[1])

    label_user.append(u_id)
    label_title.append(t_id)

label_user_set = set(label_user)
label_title_set = set(label_title)

label_user_size = len(label_user_set)
label_title_size = len(label_title_set)

label_user_max = max(list(label_user_set))
label_title_max = max(list(label_title_set))

print("label user count {}".format(label_user_size))
print("label title count {}".format(label_title_size))

user_max = max(list(user_set))
title_max = max(list(title_set))

# label 整理
pre_labels_user = np.zeros(user_max+1, dtype=float)
for item in label_list:
    u_id = int(item[0])
    t_id = int(item[1])
    pre_labels_user[u_id] = float(list(label_title_set).index(t_id))
print("---- train label 計算數量 完畢 ----")

pre_train_data = []
pre_label = []

print("user max {}".format(user_max))
print("title max {}".format(title_max))

print("---- 前處理開始 ----")
pbar = pyprind.ProgBar(len(data_list))
for item in data_list:
    t_id = int(item[2])
    u_id = int(item[1])
    watch_time = item[3]
    u_index = u_id + 1
    t_index = t_id + u_id + 1

    #data_list = (watch_time if x==0 else 1.0 if x==u_index else 1.0 if x==t_index else 0.0 for x in range(0, user_max+title_max+2))
    data_list = np.zeros(user_max+title_max+2, dtype=float)
    data_list[0] = watch_time
    data_list[u_index] = 1.0
    data_list[t_index] = 1.0

    pre_train_data.append(data_list)
    pre_label.append(pre_labels_user[u_id])
    pbar.update()

print(list(pre_train_data[0]))
print(pre_label[0])

X = pre_train_data
y = pre_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("---- 前處理結束 ----")

print("---- 訓練開始 ----")
clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

classes = [i for i in range(0, label_title_max+1)]

clf.fit(X_train, y_train)
print("Training set score: %f" % clf.score(X_test, y_test))
print("Training set loss: %f" % clf.loss_)

print("---- 訓練結束 ----")
