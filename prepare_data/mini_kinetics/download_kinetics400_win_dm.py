import os
import urllib.request
import shutil

# Donwloading was done by using links and adapting scripts from this repository:
# https://github.com/cvdfoundation/kinetics-dataset

path_files_dir = r'D:\kinetics\400'
path_filenames = ['k400_test_path.txt', 'k400_train_path.txt', 'k400_val_path.txt']
category_names = ['test', 'train', 'val']
windows_root_dir = r'D:\kinetics'

for cat_ind, cat_name in enumerate(category_names):
    target_cat_dir_path = os.path.join(windows_root_dir, cat_name)
    if not os.path.exists(target_cat_dir_path):
        os.makedirs(target_cat_dir_path)
    cat_paths_file_path = os.path.join(path_files_dir, path_filenames[cat_ind])
    f = open(cat_paths_file_path, "r")
    web_file_paths = f.readlines()
    f.close()
    web_file_paths = [file_path.strip() for file_path in web_file_paths]

    num_of_src_files = len(web_file_paths)
    for web_path_ind, web_path in enumerate(web_file_paths):
        print('%s: downloding tar file %d of %d' % (cat_name, web_path_ind, num_of_src_files))
        src_filename = web_path.split('/')[-1]
        windows_path = os.path.join(target_cat_dir_path, src_filename)
        with urllib.request.urlopen(web_path) as response, open(windows_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
