import numpy as np
import cv2
import yaml
import time

def read_calib(filepath):
  calib_params = dict()

  fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
  calib_params[fs.getNode("modality-1").string()] = dict(
    camera_matrix = fs.getNode("camera_matrix_1").mat(),
    dist_coeffs = fs.getNode("dist_coeffs_1").mat(),
  )
  calib_params[fs.getNode("modality-2").string()] = dict(
    camera_matrix = fs.getNode("camera_matrix_2").mat(),
    dist_coeffs = fs.getNode("dist_coeffs_2").mat(),
    R = fs.getNode("R").mat(),
    t = fs.getNode("T").mat()
  )
  fs.release()

  return calib_params


def camera_matrix_to_intrinsics(K):
  '''

  fx = K[0,0]
  fy = K[1,1]
  cx = K[0,2]
  cy = K[1,2]

  :param K: camera matrix
  :return: (fx, fy), (cx, cy)
  '''
  return (K[0,0], K[1,1]), (K[0,2], K[1,2])


def align_to_depth_fast(D, Kd, Ko, scale_d, R, t):
  (fx_d, fy_d), (cx_d, cy_d) = camera_matrix_to_intrinsics(Kd.astype(np.float32))
  (fx_o, fy_o), (cx_o, cy_o) = camera_matrix_to_intrinsics(Ko.astype(np.float32))

  # Reshape flattens this way:
  # [[1,2],
  # [3,4],   -->   [1,2,3,4,5,6]
  # [5,6]]
  i = np.repeat(np.linspace(0, D.shape[0]-1, D.shape[0], dtype=np.float32), D.shape[1])
  j = np.tile(np.linspace(0, D.shape[1]-1, D.shape[1], dtype=np.float32), D.shape[0])
  d = np.reshape(D, [np.prod(D.shape),]).astype(np.float32)

  z = d * scale_d
  x = ((j - cx_d) * z) / fx_d
  y = ((i - cy_d) * z) / fy_d

  P = np.concatenate([x[np.newaxis,:], y[np.newaxis,:], z[np.newaxis,:]], axis=0)

  P_t = np.matmul(R.astype(np.float32), P) + t.astype(np.float32)

  map_x = np.reshape(P_t[0, :] * fx_o / P_t[2, :] + cx_o, D.shape)
  map_y = np.reshape(P_t[1, :] * fy_o / P_t[2, :] + cy_o, D.shape)

  return map_x, map_y


def align_to_depth_slow(D, Kd, Ko, scale_d, R, t):
  (fx_d, fy_d), (cx_d, cy_d) = camera_matrix_to_intrinsics(Kd)
  (fx_o, fy_o), (cx_o, cy_o) = camera_matrix_to_intrinsics(Ko)

  map_x = (-1) * np.ones(D.shape, dtype=np.float32)
  map_y = (-1) * np.ones(D.shape, dtype=np.float32)

  for i in range(D.shape[0]):
    for j in range(D.shape[1]):
      z = scale_d * D[i,j]
      # if z > 0:
      x = (j - cx_d) * z / fx_d
      y = (i - cy_d) * z / fy_d

      p_x = R[0,0] * x + R[0,1] * y + R[0,2] * z + t[0]
      p_y = R[1,0] * x + R[1,1] * y + R[1,2] * z + t[1]
      p_z = R[2,0] * x + R[2,1] * y + R[2,2] * z + t[2]

      map_x[i,j] = p_x * fx_o / p_z + cx_o
      map_y[i,j] = p_y * fy_o / p_z + cy_o

  return map_x, map_y


if __name__ == "__main__":

  # Load data

  fs = cv2.FileStorage("data/2019-05-07_13.43.07/extrinsics-parameters.calibration-blue-small.yml", flags=cv2.FILE_STORAGE_READ)
  K_d = fs.getNode("camera_matrix_1").mat()
  K_o = fs.getNode("camera_matrix_2").mat()
  distcoeffs_d = fs.getNode("dist_coeffs_1").mat()
  distcoeffs_o = fs.getNode("dist_coeffs_2").mat()
  R_to_d = fs.getNode("R").mat()
  T_to_d = fs.getNode("T").mat()
  fs.release()

  fs = cv2.FileStorage("data/2019-05-07_13.43.07/rs_info.yml", flags=cv2.FILE_STORAGE_READ)
  scale_d = fs.getNode("depth_scale").real()
  fs.release()

  I_d_p = cv2.imread("data/2019-05-07_13.43.07/rs/depth/d_00019391.png", cv2.IMREAD_UNCHANGED)
  I_t = cv2.imread("data/2019-05-07_13.43.07/pt/thermal/t_00019391.png", cv2.IMREAD_UNCHANGED)

  I_d = cv2.normalize(I_d_p, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  I_d = cv2.applyColorMap(I_d, cv2.COLORMAP_BONE)

  I_t = cv2.normalize(I_t, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  I_t = cv2.resize(I_t, (960, 720))
  I_t = cv2.applyColorMap(I_t, cv2.COLORMAP_JET)
  padding = int((1280-960)/2)
  I_t = cv2.copyMakeBorder(I_t, 0, 0, padding, padding, 0)

  I_d_p = cv2.undistort(I_d_p, K_d, distcoeffs_d)
  I_t = cv2.undistort(I_t, K_o, distcoeffs_o)

  st_time = time.time()
  map_x_fast, map_y_fast = align_to_depth_fast(I_d_p, K_d, K_o, scale_d, R_to_d, T_to_d)
  print(f"align_to_depth fast took {time.time() - st_time} secs.")

  # st_time = time.time()
  # map_x_slow, map_y_slow = align_to_depth_slow(I_d_p, K_d, K_o, scale_d, R_to_d, T_to_d)
  # print(f"align_to_depth slow took {time.time() - st_time} secs.")

  # assert np.allclose(map_x_slow, map_x_fast) and np.allclose(map_y_slow, map_y_fast)
  map_x = map_x_fast
  map_y = map_y_fast

  st_time = time.time()
  I_t = cv2.remap(I_t, map_x, map_y, cv2.INTER_LINEAR)
  print(f"cv2.remap took {time.time() - st_time} secs.")

  cv2.imshow("window", I_t)
  cv2.waitKey()