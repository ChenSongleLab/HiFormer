
import os
import argparse
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description='Code for zip uploading.')
    parser.add_argument("--local-dir", type=str, help="the name of local dir.")
    parser.add_argument("--dst-dir", type=str, help="the name of hdfs dir.")
    args = parser.parse_args()
    return args


def file_uploading(local_dir, dst_dir):

    local_model = os.path.join(local_dir, "checkpoint_best.pth.tar")
    local_log = os.path.join(local_dir, "train_log.txt")
    local_event = os.path.join(local_dir, "event*")
    cmd_model = "hadoop fs -copyFromLocal -f " + local_model + " " + dst_dir
    cmd_log = "hadoop fs -copyFromLocal -f " + local_log + " " + dst_dir
    cmd_event = "hadoop fs -copyFromLocal -f " + local_event + " " + dst_dir
    os.system(cmd_model)
    os.system(cmd_log)
    os.system(cmd_event)


if __name__ == "__main__":
    args = parse_args()
    file_uploading(args.local_dir, args.dst_dir)

