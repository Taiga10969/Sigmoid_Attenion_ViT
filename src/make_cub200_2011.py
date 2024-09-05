import argparse
import json
import random

from tqdm import tqdm

parser = argparse.ArgumentParser(description='make_cub200_2011')
parser.add_argument('--file_path', type=str, default="./datasets/CUB_200_2011/images.txt")
parser.add_argument('--N', type=float, default=0.8)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()


with open(args.file_path, 'r') as f:
    lines = f.readlines()

label_list = []
for line in tqdm(lines):
    parts = line.split(' ', 1)
    class_id = parts[1].split('/')[0]
    
    if class_id not in label_list:
        label_list.append(class_id)

print("label num len(label_list) : ", len(label_list))

dataset = {}
for line in tqdm(lines):
    parts = line.split(' ', 1)
    class_id = parts[1].split('/')[0]

    label = label_list.index(class_id)
    
    if class_id not in dataset:
        dataset[class_id] = []
    dataset[class_id].append([parts[1].strip(), label, parts[0].strip()])


train_data=[]
test_data=[]
for key in tqdm(list(dataset.keys())):
    class_datas = dataset[key]
    class_data_num = len(class_datas)

    if args.shuffle:
        random.shuffle(class_datas)

    split_index = int(class_data_num * args.N)

    train_data += class_datas[:split_index]
    test_data += class_datas[split_index:]

print("len(train_data) : ", len(train_data))
print("len(test_data) : ", len(test_data))

CUB_DATASET={"train":train_data,
             "test":test_data,
             "label_list":label_list
             }

# JSONファイルに保存
with open("./datasets/cub200_2011_dataset.json", "w") as json_file:
    json.dump(CUB_DATASET, json_file, ensure_ascii=False, indent=4)
print("make cub_200_2011 dataset successful.")
