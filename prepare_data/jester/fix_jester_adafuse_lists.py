import os
import numpy as np

# -------------------------------------------------------
db_root_path = '/stage/algo-datasets/DB/CBI/Jester'

adafuse_src_train_list_path = '/home/yonatand/cbi/AdaFuse/data/jester/train_split.txt'
adafuse_fixed_train_list_path = '/home/yonatand/cbi/AdaFuse/data/jester/train_split_nov21.txt'
train_dir_name = 'Train'

adafuse_src_val_list_path = '/home/yonatand/cbi/AdaFuse/data/jester/validation_split.txt'
adafuse_fixed_val_list_path = '/home/yonatand/cbi/AdaFuse/data/jester/validation_split_nov21.txt'
val_dir_name = 'Validation'

# -------------------------------------------------------


def fix_db_filelist(_db_root_path, src_list_path, fixed_list_path, db_dir_name):

    file_src = open(src_list_path, 'r')
    file_fixed = open(fixed_list_path, 'w')

    src_lines = 0
    fixed_lines = 0
    while True:
        src_line = file_src.readline()

        if not src_line:
            break
        clip_name = src_line.split(' ')[0]
        if os.path.exists(os.path.join(_db_root_path, db_dir_name, clip_name)):
            file_fixed.write(src_line)
            fixed_lines += 1
        src_lines += 1

    file_src.close()
    file_fixed.close()

    print('Num of files in %s: %d' % (src_list_path, src_lines))
    print('Num of files in %s: %d' % (fixed_list_path, fixed_lines))



if __name__ == "__main__":

    fix_db_filelist(db_root_path, adafuse_src_train_list_path, adafuse_fixed_train_list_path, train_dir_name)
    fix_db_filelist(db_root_path, adafuse_src_val_list_path, adafuse_fixed_val_list_path, val_dir_name)