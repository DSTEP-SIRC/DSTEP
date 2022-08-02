import os
import re

# -------------------------------------------------------
db_root_path = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_frames'

adafuse_src_train_list_path = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_train_videofolder.txt'
adafuse_fixed_train_list_path = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_train_videofolder_nov21.txt'
train_orig_prefix = 'images/train/'
train_new_prefix = 'train/'

adafuse_src_val_list_path = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_val_videofolder.txt'
adafuse_fixed_val_list_path = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_val_videofolder_nov21.txt'
val_orig_prefix = 'images/val/'
val_new_prefix = 'val/'

# -------------------------------------------------------


def fix_db_filelist(_db_root_path, src_list_path, fixed_list_path, src_prefix, new_prefix):

    file_src = open(src_list_path, 'r')
    file_fixed = open(fixed_list_path, 'w')

    src_lines = 0
    fixed_lines = 0
    while True:
        src_line = file_src.readline()

        if not src_line:
            break
        space_positions = [m.start() for m in re.finditer(' ', src_line)]
        clip_name = src_line[:space_positions[-2]]
        clip_name = clip_name.replace(src_prefix, new_prefix)
        if os.path.exists(os.path.join(_db_root_path, clip_name)):
            new_line = src_line.replace(src_prefix, new_prefix)
            # space_positions = [m.start() for m in re.finditer(' ', new_line)]
            # orig_clip_str = src_line[:space_positions[-2]]
            # new_clip_str = '"{}"'.format(orig_clip_str.replace(src_prefix, new_prefix))
            # new_line = new_clip_str + new_line[space_positions[-2]:]
            file_fixed.write(new_line)
            fixed_lines += 1
        src_lines += 1

    file_src.close()
    file_fixed.close()

    print('Num of files in %s: %d' % (src_list_path, src_lines))
    print('Num of files in %s: %d' % (fixed_list_path, fixed_lines))



if __name__ == "__main__":

    fix_db_filelist(db_root_path, adafuse_src_train_list_path, adafuse_fixed_train_list_path,
                    train_orig_prefix, train_new_prefix)
    # fix_db_filelist(db_root_path, adafuse_src_val_list_path, adafuse_fixed_val_list_path,
    #                 val_orig_prefix, val_new_prefix)