import os
import shutil
from tqdm import tqdm
import re

# Donwloading was done by using links and adapting scripts from this repository:
# https://github.com/cvdfoundation/kinetics-dataset

src_root_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400'
src_splits_names = ['test', 'train', 'val']
corrupted_replacement_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400/replacement_for_corrupted_k400'
corrupted_list_file = '/stage/algo-datasets/DB/CBI/Kinetics_400/replacement_for_corrupted_k400/corrupted_list.txt'

mini_kinetics_train_file = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_train_videofolder.txt'
mini_kinetics_train_split_prefix = 'images/train/'
mini_kinetics_val_file = '/home/yonatand/cbi/AdaFuse/data/kinetics/mini_val_videofolder.txt'
mini_kinetics_val_split_prefix = 'images/val/'
target_train_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/train'
target_val_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/val'


def filter_split_mp4s(minik_split_file, minik_split_prefix, minik_target_folder):

    f = open(minik_split_file, 'r')
    minik_split_file_lines = f.readlines()
    f.close()
    split_clip_names = []
    split_cat_name = []
    for target_split_file_line in minik_split_file_lines:
        space_positions = [m.start() for m in re.finditer(' ', target_split_file_line)]
        target_split_filename = target_split_file_line[:space_positions[-2]]
        assert target_split_filename.startswith(minik_split_prefix)
        target_split_cat_name_clip_name = target_split_filename.replace(minik_split_prefix, '')
        cat_name, clip_name = target_split_cat_name_clip_name.split('/')
        split_clip_names.append(clip_name)
        split_cat_name.append(cat_name)

    # read corrputed file list
    f = open(corrupted_list_file, 'r')
    corrupted_mp4s_list = f.readlines()
    f.close()
    corrupted_mp4s_list = [file.strip() for file in corrupted_mp4s_list]

    total_clips_copied = 0
    orig_clips_copied = 0
    corrupted_replacement_copied = 0

    print('Filtering clips for :%s' % minik_split_prefix)
    for minik_clip_ind, minik_clip_name in enumerate(tqdm(split_clip_names)):
        minik_clip_cat_name = split_cat_name[minik_clip_ind]

        src_clip_found = False
        try:
            corrputed_ind = corrupted_mp4s_list.index(minik_clip_name + '.mp4')
            fixed_file_path = os.path.join(corrupted_replacement_dir, corrupted_mp4s_list[corrputed_ind])
            if os.path.exists(fixed_file_path):
                target_dir_path = os.path.join(minik_target_folder, minik_clip_cat_name)
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)
                shutil.move(fixed_file_path, target_dir_path)

                corrupted_replacement_copied += 1
                total_clips_copied += 1

            src_clip_found = True
        except:
            pass

        src_split_ind = 0
        while not(src_clip_found) and src_split_ind < len(src_splits_names):
            possible_clip_path = os.path.join(src_root_dir, src_splits_names[src_split_ind], minik_clip_name + '.mp4')
            if os.path.exists(possible_clip_path):
                target_dir_path = os.path.join(minik_target_folder, minik_clip_cat_name)
                if not os.path.exists(target_dir_path):
                    os.makedirs(target_dir_path)
                shutil.move(possible_clip_path, target_dir_path)

                orig_clips_copied += 1
                total_clips_copied += 1
                src_clip_found = True
                break
            src_split_ind += 1

        if not src_clip_found:
            print('Not found: %s - %s' % (minik_clip_cat_name, minik_clip_name))
    print('After filtering copied %d clips (%d original, %d corrupted files replaced)' %
          (total_clips_copied, orig_clips_copied, corrupted_replacement_copied))

    return True


if __name__ == "__main__":

    # filter validation mp4s
    #filter_split_mp4s(mini_kinetics_val_file, mini_kinetics_val_split_prefix, target_val_dir)

    # filter train mp4s
    #filter_split_mp4s(mini_kinetics_train_file, mini_kinetics_train_split_prefix, target_train_dir)