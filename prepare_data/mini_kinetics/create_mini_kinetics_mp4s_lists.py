import os

train_mp4s_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/train'
train_mp4s_list = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/train_mp4s.txt'
val_mp4s_dir = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/val'
val_mp4s_list = '/stage/algo-datasets/DB/CBI/Kinetics_400/mini_kinetics_mp4s/val_mp4s.txt'


def write_videos_rel_paths_list(videos_dir, videos_list_file):
    videos_full_list = []
    cats_list = os.listdir(videos_dir)
    for cat_ind, cat_name in enumerate(cats_list):
        print('listing videos, category %d of %d' % (cat_ind, len(cats_list)))
        cat_videos_path = os.path.join(videos_dir, cat_name)
        videos_list = os.listdir(cat_videos_path)
        for video_filename in videos_list:
            if video_filename.endswith('.mp4'):
                videos_full_list.append(os.path.join(cat_name, video_filename))

    videos_full_list = [file_path + '\n' for file_path in videos_full_list]
    f = open(videos_list_file, 'w')
    f.writelines(videos_full_list)
    f.close()



if __name__ == "__main__":

    # write validation mp4 paths to file
    #write_videos_rel_paths_list(val_mp4s_dir, val_mp4s_list)

    # write train mp4 paths to file
    write_videos_rel_paths_list(train_mp4s_dir, train_mp4s_list)


