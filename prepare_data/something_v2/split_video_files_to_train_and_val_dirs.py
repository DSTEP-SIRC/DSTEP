import os
import re
import shutil
from tqdm import tqdm


all_clips_dir = '/stage/algo-datasets/DB/CBI/something2something-v2/frames/20bn-something-something-v2'
target_train_clips_dir = '/stage/algo-datasets/DB/CBI/something2something-v2/frames/train_val_webm/train'
target_val_clips_dir = '/stage/algo-datasets/DB/CBI/something2something-v2/frames/train_val_webm/val'

adafuse_train_gt_file = '/home/yonatand/cbi/AdaFuse/data/somethingv2/train_videofolder.txt'
adafuse_val_gt_file = '/home/yonatand/cbi/AdaFuse/data/somethingv2/val_videofolder.txt'

write_file_list = True

def split_video_files_and_print_list(src_clips_dir, target_clips_dir, split_str, gt_file_path, print_file_list):

    adafuse_gt_file_split_prefix = split_str + '/'

    f = open(gt_file_path, 'r')
    split_file_lines = f.readlines()
    f.close()

    files_in_adafuse_gt = 0
    files_copied = 0
    videos_full_list = []
    for target_split_file_line in split_file_lines:
        space_positions = [m.start() for m in re.finditer(' ', target_split_file_line)]
        target_split_filename = target_split_file_line[:space_positions[-2]]
        assert target_split_filename.startswith(adafuse_gt_file_split_prefix)
        cur_clip_name = target_split_filename.replace(adafuse_gt_file_split_prefix, '')
        cur_clip_filename = cur_clip_name + '.webm'
        cur_clip_src_path = os.path.join(src_clips_dir, cur_clip_filename)

        files_in_adafuse_gt += 1
        if os.path.exists(cur_clip_src_path):
            shutil.move(cur_clip_src_path, target_clips_dir)
            videos_full_list.append(cur_clip_filename)
            files_copied += 1

    print('Files in adafuse GT file:       %d' % files_in_adafuse_gt)
    print('Actual files copied to new dir: %d' % files_copied)

    if print_file_list:
        videos_list_file = target_clips_dir + '_clips.txt'
        videos_full_list = [file_path + '\n' for file_path in videos_full_list]
        f = open(videos_list_file, 'w')
        f.writelines(videos_full_list)
        f.close()

    return True


if __name__ == "__main__":

    # split validation clips
    #split_video_files_and_print_list(all_clips_dir, target_val_clips_dir, 'val', adafuse_val_gt_file, write_file_list)

    # split train clips
    #split_video_files_and_print_list(all_clips_dir, target_train_clips_dir, 'train', adafuse_train_gt_file, write_file_list)