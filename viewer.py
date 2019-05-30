import synchronization as sync
import calibration as calib
from os.path import join, basename
from collections import OrderedDict
import glob
import csv
import re
import ast
import cv2
import numpy as np
import time
import argparse

CONSTANTS = dict(
  color_dir   = "rs/color/",
  depth_dir   = "rs/depth/",
  thermal_dir = "pt/thermal/",
  color_prefix = "c_",
  depth_prefix = "d_",
  thermal_prefix = "t_",
  color_fext   = ".jpg",
  depth_fext   = ".png",
  thermal_fext = ".png",
  rs_log_file = "rs.log",
  pt_log_file = "pt.log",
  calibration_file = "extrinsics-parameters.calibration-blue-small.yml",  # TODO: change to calibration.yml and include ``calibration-blue-small'' as a variable in it
  rs_info_file = "rs_info.yml",
  image_dims = (1280, 720)
)

def parse_fid_from_filename(filename):
  from os.path import splitext
  return splitext(basename(filename))[0].split("_")[-1]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str, default="data/2019-05-07_13.43.07/", help="Input path of the sequence to view")
  opt = parser.parse_args()

  # filter annotations corresponding to sequence referenced by ''opt.input_path''

  annot_files = glob.glob("data/human-detections.faster_rcnn_nas_coco_2018_01_28.*.csv")

  boxes_all = OrderedDict()
  target_sqn_name = basename(opt.input_path.strip('/'))
  for annot_f in annot_files:
    with open(annot_f, 'r') as f:
      reader = csv.DictReader(f)
      for line in reader:
        fp = line['filepath']
        match = re.search(r'(\d+-\d+-\d+_\d+.\d+.\d+)',fp)
        if match:
          sqn_name = match.group(1)
          if sqn_name == target_sqn_name:
            frame_rel_path = fp[match.span(1)[0]:]
            box_pair = (ast.literal_eval(line['box']), float(line['score']))
            boxes_all.setdefault(parse_fid_from_filename(frame_rel_path), []).append(box_pair)

  # get temporal synchronization timestamps of both rs's streams and pt's

  log_rs = sync.read_log_file(join(opt.input_path, "rs.log"))
  log_pt = sync.read_log_file(join(opt.input_path, "pt.log"))
  log_synced = sync.time_sync(log_rs, log_pt)

  # read calibration parameters and depth scale

  calib_params = calib.read_calib(join(opt.input_path, CONSTANTS["calibration_file"]))

  fs = cv2.FileStorage(join(opt.input_path, CONSTANTS["rs_info_file"]), flags=cv2.FILE_STORAGE_READ)
  scale_d = fs.getNode("depth_scale").real()
  fs.release()

  # visualize frames and boxes

  for frame_rs, frame_pt in log_synced:
    I_c = cv2.imread(join(opt.input_path, CONSTANTS["color_dir"], CONSTANTS["color_prefix"] + frame_rs.id + CONSTANTS["color_fext"]), cv2.IMREAD_UNCHANGED)
    I_d = cv2.imread(join(opt.input_path, CONSTANTS["depth_dir"], CONSTANTS["depth_prefix"] + frame_rs.id + CONSTANTS["depth_fext"]), cv2.IMREAD_UNCHANGED)
    I_t = cv2.imread(join(opt.input_path, CONSTANTS["thermal_dir"], CONSTANTS["thermal_prefix"] + frame_rs.id + CONSTANTS["thermal_fext"]), cv2.IMREAD_UNCHANGED)

    # I_d3 = cv2.normalize(I_d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # I_d3 = cv2.applyColorMap(I_d3, cv2.COLORMAP_BONE)

    I_t = cv2.normalize(I_t, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    I_t = cv2.resize(I_t, (int(float(I_t.shape[1])/I_t.shape[0] * CONSTANTS["image_dims"][1]), CONSTANTS["image_dims"][1]))
    I_t = cv2.applyColorMap(I_t, cv2.COLORMAP_JET)
    padding = int((CONSTANTS["image_dims"][0] - I_t.shape[1]) / 2)
    I_t = cv2.copyMakeBorder(I_t, 0, 0, padding, padding, 0)

    K_c = K_d = calib_params["Color"]["camera_matrix"]  # K_c == K_d (given the internal calibration of both modalities in realsense's D435)
    K_t = calib_params["Thermal"]["camera_matrix"]

    I_d = cv2.undistort(I_d, K_d, calib_params["Color"]["dist_coeffs"])
    I_t = cv2.undistort(I_t, K_t, calib_params["Thermal"]["dist_coeffs"])

    map_x, map_y = calib.align_to_depth_fast(I_d, K_d, K_t, scale_d, calib_params["Thermal"]["R"], calib_params["Thermal"]["t"])
    I_t = cv2.remap(I_t, map_x, map_y, cv2.INTER_LINEAR)

    I_d = cv2.normalize(I_d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    I_d = cv2.applyColorMap(I_d, cv2.COLORMAP_BONE)

    # draw boxes

    if frame_rs.id in boxes_all:
      boxes = boxes_all[frame_rs.id]
      for box, score in boxes:
        cv2.rectangle(I_c, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
        cv2.rectangle(I_d, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
        cv2.rectangle(I_t, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)

    cv2.imshow("color", I_c)
    cv2.imshow("depth", I_d)
    cv2.imshow("thermal", I_t)
    cv2.waitKey(33)

  quit()