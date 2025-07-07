import os
import shutil

project_root_dir = "/Change/it/to/your/path/ChineseDigitRecognition/"
data_pic_dir = "data/data"
pic_dir = os.path.join(project_root_dir, data_pic_dir)
pics = os.listdir(pic_dir)

if os.path.exists(os.path.join(pic_dir, "train_pic")) or os.path.exists(
    os.path.join(pic_dir, "test_pic")
):
    print("path exists")
else:
    print("path dosen't exists")
    os.mkdir(os.path.join(pic_dir, "train_pic"))
    os.mkdir(os.path.join(pic_dir, "test_pic"))

train_dataset = os.path.join(pic_dir, "train_pic")
test_dataset = os.path.join(pic_dir, "test_pic")
train_pics = pics[3002:]
test_pics = pics[:3002]

for train_pic in train_pics:
    pic_current_dir = os.path.join(pic_dir, train_pic)
    shutil.move(pic_current_dir, train_dataset)

for test_pic in train_pics:
    pic_current_dir = os.path.join(pic_dir, test_pic)
    shutil.move(pic_current_dir, test_dataset)
