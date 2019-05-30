from os.path import join

class Timestamp:
  def __init__(self, id, time):
    self.id = id
    self.time = time

def read_log_file(filepath):
  log = []
  with open(filepath, 'r') as f:
    lines = f.readlines()
    for l in lines:
      id, time = l.strip().split(',')
      log.append(Timestamp(id, int(time)))

  return log

def time_sync(a, b, eps = 50, verbose=False):
  ab = []

  master, slave = (a, b) if len(a) < len(b) else (b, a)

  j = 0

  for i, ts_m in enumerate(master):
    matches = []
    while ((j < len(slave)) and (slave[j].time < (ts_m.time + eps))):
      dist = abs(ts_m.time - slave[j].time)
      if dist < eps:
        matches.append((j,dist))
      j += 1

    if len(matches) > 0:
      m_best = matches[0]
      for k in range(len(matches)):
        if matches[k][1] < m_best[1]:
          m_best = matches[k]

      ab.append( (master[i], slave[m_best[0]]) if len(a) < len(b) else (slave[m_best[0]], master[i]) )
      j = m_best[0]

  if verbose:
    diff_total = .0
    for i in range(len(ab)):
      diff = ab[i][0].time - ab[i][1].time
      print(f"{ab[i][0].id}@{ab[i][0].time}, {ab[i][1].id}@{ab[i][1].time}, {diff}")
      diff_total += abs(diff)

    print(f"Average abs diff between timestamps = {diff_total/len(ab)}")

  return ab

if __name__ == "__main__":
  data_path = "data/"
  target_sqn_name = "2019-05-07_13.43.07"

  log_rs = read_log_file(join(data_path, target_sqn_name, "rs.log"))
  log_pt = read_log_file(join(data_path, target_sqn_name, "pt.log"))
  log_synced = time_sync(log_rs, log_pt)

