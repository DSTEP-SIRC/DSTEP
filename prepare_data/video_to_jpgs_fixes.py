from __future__ import print_function, division
import os
import time
import subprocess
from tqdm import tqdm
import argparse
import sys

parser = argparse.ArgumentParser(description="Dataset processor: Video->Frames")
parser.add_argument("dir_path", type=str, help="original dataset path")
parser.add_argument("dst_dir_path", type=str, help="dest path to save the frames")
parser.add_argument("--prefix", type=str, default="image_%05d.jpg", help="output image type")
parser.add_argument("--accepted_formats", type=str, default=[".mp4", ".mkv", ".webm"], nargs="+",
                    help="list of input video formats")
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=666666666)
parser.add_argument("--file_list", type=str, default="")
parser.add_argument("--frame_rate", type=int, default=-1)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()


def inner_main(argv):
    t0 = time.time()
    dir_path = args.dir_path
    dst_dir_path = args.dst_dir_path

    if args.file_list == "":
        file_names = sorted(os.listdir(dir_path))
    else:
        file_names = [x.strip() for x in open(args.file_list).readlines()]
    del_list = []
    for i, file_name in enumerate(file_names):
        if not any([x in file_name for x in args.accepted_formats]):
            del_list.append(i)
    file_names = [x for i, x in enumerate(file_names) if i not in del_list]
    file_names = file_names[args.begin:args.end + 1]
    print("%d videos to handle (after %d being removed)" % (len(file_names), len(del_list)))

    num_of_fixed_clips = 0
    num_of_empty_clips = 0
    for file_name in tqdm(file_names):

        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_dir_path, name)

        video_file_path = '"{}"'.format(os.path.join(dir_path, file_name))
        if not os.path.exists(dst_directory_path):
            os.makedirs(dst_directory_path, exist_ok=True)
        else:
            if os.path.exists(os.path.join(dst_directory_path, args.prefix % 1)):
                continue

        if args.frame_rate > 0:
            frame_rate_str = "-r %d" % args.frame_rate
        else:
            frame_rate_str = ""
        cmd = 'ffmpeg -nostats -loglevel 0 -i {} -vf scale=-1:360 {} "{}/{}"'.format(video_file_path, frame_rate_str,
                                                                                   dst_directory_path, args.prefix)
        if args.dry_run:
            print(cmd)
        else:
            subprocess.call(cmd, shell=True)

        if os.path.exists(os.path.join(dst_directory_path, args.prefix % 1)):
            num_of_fixed_clips += 1
            if num_of_fixed_clips % 100 == 0:
                print('fixed %d clips' % num_of_fixed_clips)
        else:
            num_of_empty_clips += 1

    print('num of fixed clips: %d' % num_of_fixed_clips)
    print('num of empty clips: %d' % num_of_empty_clips)
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))
    os.system("stty sane")


def main(argv):
    inner_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])