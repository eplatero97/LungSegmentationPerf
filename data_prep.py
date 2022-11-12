# Import libraries
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

annot_path = 'data/selected/masks/'
image_path = 'data/selected/CXR_png/'

shutil.copytree('data/original', 'data/selected')

annot_files = os.listdir(annot_path)
annot_files.sort()
for annot in annot_files:
    name = annot
    if 'mask' in annot:
        name = annot.split('_mask')[0]+'.png'
    os.rename(annot_path+annot, annot_path+name)

image_files = os.listdir(image_path)
image_files.sort()
annot_files = os.listdir(annot_path)
annot_files.sort()
for img in image_files:
    if img not in annot_files:
        os.remove(image_path+img)

train_destination_path = 'data/processed/train/CXR_png/'
val_destination_path = 'data/processed/val/CXR_png/'
test_destination_path = 'data/processed/test/CXR_png/'
train_mask_path = 'data/processed/train/masks/'
val_mask_path = 'data/processed/val/masks/'
test_mask_path = 'data/processed/test/masks/'

os.mkdir('data/processed')
os.mkdir('data/processed/train')
os.mkdir(train_destination_path)
os.mkdir(train_mask_path)
os.mkdir('data/processed/val')
os.mkdir(val_destination_path)
os.mkdir(val_mask_path)
os.mkdir('data/processed/test')
os.mkdir(test_destination_path)
os.mkdir(test_mask_path)

num_data = len(os.listdir(annot_path))

data = [i for i in range(704)]

X_train, X_test, _, _ = train_test_split(data, data, test_size=0.2)
X_test, X_val, _, _ = train_test_split(X_test, X_test, test_size=0.5)
print(len(X_train))
print(len(X_test))
print(len(X_val))
X_train, X_val, X_test = set(X_train), set(X_val), set(X_test)

image_files = os.listdir(image_path)
image_files.sort()
annot_files = os.listdir(annot_path)
annot_files.sort()
for i in range(num_data):
    if i in X_train:
        shutil.copyfile(image_path+image_files[i], train_destination_path+image_files[i])
        shutil.copyfile(annot_path+annot_files[i], train_mask_path+annot_files[i])
    elif i in X_test:
        shutil.copyfile(image_path+image_files[i], test_destination_path+image_files[i])
        shutil.copyfile(annot_path+annot_files[i], test_mask_path+annot_files[i])
    elif i in X_val:
        shutil.copyfile(image_path+image_files[i], val_destination_path+image_files[i])
        shutil.copyfile(annot_path+annot_files[i], val_mask_path+annot_files[i])