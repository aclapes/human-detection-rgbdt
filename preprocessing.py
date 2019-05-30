import numpy as np
import csv
import glob
import re
from os.path import join, splitext
import cv2
from collections import OrderedDict
import ast
import time
import synchronization as sync


# if __name__ == "__main__":
  # data_path = "data/"
  # target_sqn_name = "2019-05-07_13.43.07"
  # md_prefix = "pt/thermal/"
  # file_ext = ".png"
  # im_dims = (1280,720)
  #
  # log_rs = sync.read_log_file(join(data_path, target_sqn_name, "rs.log"))
  # log_pt = sync.read_log_file(join(data_path, target_sqn_name, "pt.log"))
  # log_synced = sync.time_sync(log_rs, log_pt)
  #
  # annot_files = glob.glob("data/human-detections.faster_rcnn_nas_coco_2018_01_28.*.csv")
  #
  # boxes_all = OrderedDict()
  # for annot_f in annot_files:
  #   with open(annot_f, 'r') as f:
  #     reader = csv.DictReader(f)
  #     for line in reader:
  #       fp = line['filepath']
  #       match = re.search(r'(\d+-\d+-\d+_\d+.\d+.\d+)',fp)
  #       if match:
  #         actual_sqn_name = match.group(1)
  #         if actual_sqn_name == target_sqn_name:
  #           frame_rel_path = fp[match.span(1)[0]:]
  #           box_pair = (ast.literal_eval(line['box']), float(line['score']))
  #           boxes_all.setdefault(parse_fid_from_filename(frame_rel_path), []).append(box_pair)
  #
  # target_frames_re = join(md_prefix, "*" + file_ext)
  #
  # frame_paths = glob.glob(join(data_path, target_sqn_name, target_frames_re))
  # for fp in sorted(frame_paths):
  #   im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
  #   st_time = time.time()
  #   im = cv2.resize(im, im_dims)
  #   print(f"Elapsed time: {time.time() - st_time} secs.")
  #   if len(im.shape) == 2:
  #       if im.dtype == "uint16": # assume depth map
  #         im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  #       if im.dtype == "uint8":
  #         im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
  #
  #   key = parse_fid_from_filename(fp)
  #   if key in boxes_all:
  #     boxes = boxes_all[key]
  #     for box, score in boxes:
  #       cv2.rectangle(im, (box[1], box[0]), (box[3], box[2]), (0,0,255), 2)
  #
  #   cv2.imshow("window", im)
  #   cv2.waitKey(33)


