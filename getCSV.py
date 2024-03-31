import csv
import pandas as pd
import numpy as np


anno_list = pd.read_excel("path_to\\action_recognition\\AR_metadata.xlsx",sheet_name="AR") #获取表格文件

num_train_video,num_test_video = 0,0
print(f"len of xlsx is {len(anno_list)}")
for index, row in anno_list.iterrows():
    if row["type"] == "train":
        num_train_video += 1
    if row["type"] == "test":
        num_test_video += 1

num_train_samples = 8000
num_test_samples = 2000
print(num_test_video)
print(num_train_video)

#### clips in AR_metadata.xlsx, No.1 - No.24004 clip are for training, No.24005 - end are for testing. 
#### 8000 clips are fetched randomly from training set and 2000 clips are fetch randomly from testing set.
train_samples = np.random.choice(num_train_video,num_train_samples,replace=False) 
test_index_range = np.arange(num_train_video+1,len(anno_list)+1)
test_samples = np.random.choice(test_index_range,num_test_samples,replace=False)
tmp_1,tmp_2 = [],[]
with open("path_to\\action_recognition\\annotation\\train_v5.csv",'w',newline='',encoding='utf-8') as train_v1:
    csv_writer1 = csv.writer(train_v1)
    for index, row in anno_list.iterrows():

        if index in train_samples:
            video_id = row["video_id"] + ".mp4"
            label = row["labels"].split(",")[0] # take one label for each clip
            info = ["path_to/action_recognition/dataset/video/" + video_id +" " + label]
            tmp_1.append(info)

    csv_writer1.writerows(tmp_1)
with open("path_to\\action_recognition\\annotation\\test_v5.csv",'w',newline='',encoding='utf-8') as test_v1:
    csv_writer2 = csv.writer(test_v1)
    for index, row in anno_list.iterrows():
        if index in test_samples:
            
            video_id = row["video_id"] + ".mp4"
            label = row["labels"].split(",")[0]
            info = ["path_to/action_recognition/dataset/video/" + video_id +" " + label]
            tmp_2.append(info)

    csv_writer2.writerows(tmp_2)

print("Done!")