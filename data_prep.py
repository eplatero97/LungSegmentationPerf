# Import libraries
import os
import glob
import shutil
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--inputpath', type=str, required=True)
parser.add_argument('--outputpath', default='data/lungsegmentation', type=str)
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument
print('Input path:', args.inputpath, 'Output path:', args.outputpath)
if args.outputpath[-1] == '/':
    args.outputpath = args.outputpath[:-1]
annot_path = 'data/selected/masks/'
image_path = 'data/selected/CXR_png/'

if os.path.exists('data/selected/'):
    shutil.rmtree('data/selected/')
shutil.copytree(args.inputpath+'/masks', 'data/selected/masks')
shutil.copytree(args.inputpath+'/CXR_png', 'data/selected/CXR_png')

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

train_destination_path = args.outputpath+'/img_dir/train/'
val_destination_path = args.outputpath+'/img_dir/val/'
test_destination_path = args.outputpath+'/img_dir/test/'
train_mask_path = args.outputpath+'/ann_dir/train/'
val_mask_path = args.outputpath+'/ann_dir/val/'
test_mask_path = args.outputpath+'/ann_dir/test/'

if os.path.exists(args.outputpath):
    shutil.rmtree(args.outputpath)
os.mkdir(args.outputpath)
os.mkdir(args.outputpath+'/img_dir')
os.mkdir(args.outputpath+'/ann_dir')
os.mkdir(train_destination_path)
os.mkdir(train_mask_path)
os.mkdir(val_destination_path)
os.mkdir(val_mask_path)
os.mkdir(test_destination_path)
os.mkdir(test_mask_path)

num_data = len(os.listdir(annot_path))

data = [i for i in range(704)]

X_train, X_test, _, _ = train_test_split(data, data, test_size=0.2)
X_test, X_val, _, _ = train_test_split(X_test, X_test, test_size=0.5)
print('Total training items:', len(X_train))
print('Total test items:', len(X_test))
print('Total val items:', len(X_val))
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