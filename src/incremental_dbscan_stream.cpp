// Incremental DBSCAN over a dynamic (insert/delete/move) stream, used as a
// same-language (C++) field baseline for DelauCluster.
//
// The fast incremental-DBSCAN path maintains density connectivity with a grid +
// union-find over core points. Union-find is grow-only: it processes INSERTS
// exactly but cannot cheaply un-merge or demote cores, so it has no fast exact
// DELETE/MOVE. We model the fast path charitably: a deleted point is marked
// inactive and excluded from future neighbor queries, but existing union-find
// merges and core flags are left intact (the operation it structurally cannot do
// cheaply). A MOVE is a delete of the old location plus an insert of the new one.
//
// Two evaluation modes share one binary:
//   * latency (insert-only): per-op incremental vs from-scratch DBSCAN time.
//   * faithfulness (delete/move): per-op ARI of the incrementally-maintained
//     labels against a FROM-SCRATCH DBSCAN over the currently-active points,
//     using the SAME DBSCAN implementation -- so an insert-only stream yields
//     ARI 1.0 by construction (no strawman) and any drift is purely the cost of
//     the fast path's inability to process deletes/moves.
//
// With --exact, delete/move instead rebuild connectivity for the affected
// cluster from scratch: this is the strong exact incremental-DBSCAN baseline
// (ARI 1.0, no drift) whose deletion cost grows with the affected cluster and
// which still needs eps/min-pts and emits no per-update certificate.
//
// Usage:
//   incremental_dbscan_stream <base_csv> <stream_csv> <out_csv> \
//       --eps E --min-pts M [--every K]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Pt {
  double x = 0.0;
  double y = 0.0;
};

inline long long cell_key(double x, double y, double eps) {
  const long long cx = static_cast<long long>(std::floor(x / eps));
  const long long cy = static_cast<long long>(std::floor(y / eps));
  return (cx << 32) ^ (cy & 0xffffffffLL);
}

struct DSU {
  std::vector<std::size_t> parent;
  void ensure(std::size_t n) {
    while (parent.size() < n)
      parent.push_back(parent.size());
  }
  std::size_t find(std::size_t a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  }
  void unite(std::size_t a, std::size_t b) { parent[find(a)] = find(b); }
};

// Incremental DBSCAN under insert/delete/move (fast grow-only union-find path).
struct IncrementalDBSCAN {
  double eps;
  double eps2;
  int min_pts;
  std::vector<Pt> pts;
  std::vector<char> active;
  std::vector<int> cnt;   // active eps-neighbors excluding self
  std::vector<char> core;
  DSU dsu;
  std::unordered_map<long long, std::vector<std::size_t>> grid;
  std::size_t n_active = 0;
  bool exact_mode = false;  // exact delete/move via affected-cluster rebuild

  IncrementalDBSCAN(double e, int m) : eps(e), eps2(e * e), min_pts(m) {}

  std::vector<std::size_t> active_neighbors(double x, double y) const {
    std::vector<std::size_t> out;
    const long long cx = static_cast<long long>(std::floor(x / eps));
    const long long cy = static_cast<long long>(std::floor(y / eps));
    for (long long dx = -1; dx <= 1; ++dx)
      for (long long dy = -1; dy <= 1; ++dy) {
        auto it = grid.find(((cx + dx) << 32) ^ ((cy + dy) & 0xffffffffLL));
        if (it == grid.end())
          continue;
        for (std::size_t j : it->second) {
          if (!active[j])
            continue;
          const double ddx = pts[j].x - x;
          const double ddy = pts[j].y - y;
          if (ddx * ddx + ddy * ddy <= eps2)
            out.push_back(j);
        }
      }
    return out;
  }

  void insert(double x, double y) {
    const std::size_t id = pts.size();
    const std::vector<std::size_t> nbrs = active_neighbors(x, y);
    pts.push_back({x, y});
    active.push_back(1);
    cnt.push_back(static_cast<int>(nbrs.size()));
    core.push_back(0);
    dsu.ensure(pts.size());
    grid[cell_key(x, y, eps)].push_back(id);
    n_active++;

    std::vector<std::size_t> newly_core;
    if (cnt[id] + 1 >= min_pts) {
      core[id] = 1;
      newly_core.push_back(id);
    }
    for (std::size_t q : nbrs) {
      cnt[q] += 1;
      if (!core[q] && cnt[q] + 1 >= min_pts) {
        core[q] = 1;
        newly_core.push_back(q);
      }
    }
    for (std::size_t c : newly_core) {
      const auto cn = active_neighbors(pts[c].x, pts[c].y);
      for (std::size_t q : cn)
        if (q != c && core[q])
          dsu.unite(c, q);
    }
  }

  // Nearest active point to (x,y); -1 if none. Brute force (moderate n in the
  // faithfulness experiment); aligns deletes with DelauCluster's remove_nearest.
  long long nearest_active(double x, double y) const {
    long long best = -1;
    double bestd = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < pts.size(); ++i) {
      if (!active[i])
        continue;
      const double ddx = pts[i].x - x;
      const double ddy = pts[i].y - y;
      const double d = ddx * ddx + ddy * ddy;
      if (d < bestd) {
        bestd = d;
        best = static_cast<long long>(i);
      }
    }
    return best;
  }

  // Charitable fast-path delete: mark inactive + drop from future neighbor
  // queries, but do NOT update union-find / core flags / neighbor counts -- the
  // operation the grow-only fast path structurally cannot do cheaply.
  void remove_nearest(double x, double y) {
    if (exact_mode) {
      exact_remove(x, y);
      return;
    }
    const long long id = nearest_active(x, y);
    if (id < 0)
      return;
    active[id] = 0;
    n_active--;
  }

  void move_nearest(double ox, double oy, double nx, double ny) {
    remove_nearest(ox, oy);
    insert(nx, ny);
  }

  // Exact delete (Ester et al. 1998 semantics): apply the local core-flag
  // changes, then rebuild connectivity for the affected cluster(s) from scratch
  // so the labeling equals a from-scratch DBSCAN, including the split case. The
  // cost includes stored-record scans and affected-cluster neighborhood work.
  // This implementation's measurements are not a dynamic-deletion lower bound;
  // Gan & Tao (2017) instead studies from-scratch Euclidean DBSCAN.
  void exact_remove(double x, double y) {
    const long long sid = nearest_active(x, y);
    if (sid < 0)
      return;
    const std::size_t id = static_cast<std::size_t>(sid);
    std::vector<std::size_t> nbrs;
    for (std::size_t q : active_neighbors(pts[id].x, pts[id].y))
      if (q != id)
        nbrs.push_back(q);
    // Pre-deletion membership: every active core in the cluster(s) touching id.
    std::unordered_map<std::size_t, char> roots;
    if (core[id])
      roots[dsu.find(id)] = 1;
    for (std::size_t q : nbrs)
      if (core[q])
        roots[dsu.find(q)] = 1;
    std::vector<std::size_t> affected;
    for (std::size_t i = 0; i < pts.size(); ++i)
      if (active[i] && core[i] && roots.count(dsu.find(i)))
        affected.push_back(i);
    // Delete and apply local core-flag changes.
    active[id] = 0;
    n_active--;
    for (std::size_t q : nbrs) {
      cnt[q] -= 1;
      core[q] = (cnt[q] + 1 >= min_pts) ? 1 : 0;
    }
    // Rebuild connectivity among the affected cores under current adjacency
    // (deletion never connects clusters, so this is self-contained and exact).
    for (std::size_t a : affected)
      dsu.parent[a] = a;
    for (std::size_t a : affected) {
      if (!active[a] || !core[a])
        continue;
      for (std::size_t b : active_neighbors(pts[a].x, pts[a].y))
        if (b != a && active[b] && core[b])
          dsu.unite(a, b);
    }
  }

  // Incremental labels over active points (-2 for inactive). Border points adopt
  // the cluster of their smallest-index adjacent active core; otherwise noise(-1).
  std::vector<int> labels() {
    std::vector<int> lab(pts.size(), -2);
    std::unordered_map<std::size_t, int> root_to_id;
    int next = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
      if (!active[i] || !core[i])
        continue;
      const std::size_t r = dsu.find(i);
      auto it = root_to_id.find(r);
      if (it == root_to_id.end())
        it = root_to_id.emplace(r, next++).first;
      lab[i] = it->second;
    }
    for (std::size_t i = 0; i < pts.size(); ++i) {
      if (!active[i] || core[i])
        continue;
      lab[i] = -1;
      std::size_t best = std::numeric_limits<std::size_t>::max();
      for (std::size_t q : active_neighbors(pts[i].x, pts[i].y))
        if (core[q] && q < best)
          best = q;
      if (best != std::numeric_limits<std::size_t>::max())
        lab[i] = lab[best];
    }
    return lab;
  }
};

// From-scratch DBSCAN over a point set (same core/border rules as the incremental
// path) -- ground truth and latency baseline.
std::vector<int> full_dbscan(const std::vector<Pt> &pts, double eps, int min_pts) {
  const double eps2 = eps * eps;
  std::unordered_map<long long, std::vector<std::size_t>> grid;
  for (std::size_t i = 0; i < pts.size(); ++i)
    grid[cell_key(pts[i].x, pts[i].y, eps)].push_back(i);
  auto neigh = [&](std::size_t i) {
    std::vector<std::size_t> out;
    const long long cx = static_cast<long long>(std::floor(pts[i].x / eps));
    const long long cy = static_cast<long long>(std::floor(pts[i].y / eps));
    for (long long dx = -1; dx <= 1; ++dx)
      for (long long dy = -1; dy <= 1; ++dy) {
        auto it = grid.find(((cx + dx) << 32) ^ ((cy + dy) & 0xffffffffLL));
        if (it == grid.end())
          continue;
        for (std::size_t j : it->second) {
          const double ddx = pts[j].x - pts[i].x;
          const double ddy = pts[j].y - pts[i].y;
          if (ddx * ddx + ddy * ddy <= eps2)
            out.push_back(j);
        }
      }
    return out;
  };
  std::vector<std::vector<std::size_t>> nbr(pts.size());
  std::vector<char> core(pts.size(), 0);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    nbr[i] = neigh(i);
    if (static_cast<int>(nbr[i].size()) >= min_pts)
      core[i] = 1;
  }
  std::vector<int> lab(pts.size(), -1);
  int cid = 0;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    if (!core[i] || lab[i] != -1)
      continue;
    lab[i] = cid;
    std::vector<std::size_t> stack{i};
    while (!stack.empty()) {
      const std::size_t c = stack.back();
      stack.pop_back();
      for (std::size_t q : nbr[c])
        if (core[q] && lab[q] == -1) {
          lab[q] = cid;
          stack.push_back(q);
        }
    }
    ++cid;
  }
  for (std::size_t i = 0; i < pts.size(); ++i) {
    if (core[i])
      continue;
    std::size_t best = std::numeric_limits<std::size_t>::max();
    for (std::size_t q : nbr[i])
      if (core[q] && q < best)
        best = q;
    if (best != std::numeric_limits<std::size_t>::max())
      lab[i] = lab[best];
  }
  return lab;
}

// Adjusted Rand Index over a given index subset.
double adjusted_rand_index(const std::vector<int> &a, const std::vector<int> &b,
                           const std::vector<std::size_t> &idx) {
  std::unordered_map<long long, long long> nij;
  std::unordered_map<int, long long> ai, bj;
  long long n = 0;
  for (std::size_t k : idx) {
    const int x = a[k], y = b[k];
    nij[(static_cast<long long>(x) << 32) ^ (static_cast<long long>(y) & 0xffffffffLL)]++;
    ai[x]++;
    bj[y]++;
    n++;
  }
  auto c2 = [](long long m) { return static_cast<double>(m) * (m - 1) / 2.0; };
  double sum_nij = 0, sum_ai = 0, sum_bj = 0;
  for (auto &kv : nij)
    sum_nij += c2(kv.second);
  for (auto &kv : ai)
    sum_ai += c2(kv.second);
  for (auto &kv : bj)
    sum_bj += c2(kv.second);
  const double cn = c2(n);
  if (cn == 0)
    return 1.0;
  const double expected = sum_ai * sum_bj / cn;
  const double maxidx = 0.5 * (sum_ai + sum_bj);
  if (maxidx - expected == 0.0)
    return 1.0;
  return (sum_nij - expected) / (maxidx - expected);
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> f;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ','))
    f.push_back(tok);
  return f;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: incremental_dbscan_stream <base_csv> <stream_csv> "
                 "<out_csv> --eps E --min-pts M [--every K] [--exact]\n";
    return 2;
  }
  const std::string base_csv = argv[1];
  const std::string stream_csv = argv[2];
  const std::string out_csv = argv[3];
  double eps = 0.0;
  int min_pts = 4;
  int every = 1;
  bool exact = false;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--eps" && i + 1 < argc)
      eps = std::stod(argv[++i]);
    else if (a == "--min-pts" && i + 1 < argc)
      min_pts = std::stoi(argv[++i]);
    else if (a == "--every" && i + 1 < argc)
      every = std::max(1, std::stoi(argv[++i]));
    else if (a == "--exact")
      exact = true;
  }
  if (eps <= 0.0) {
    std::cerr << "error: --eps must be positive\n";
    return 2;
  }

  IncrementalDBSCAN inc(eps, min_pts);
  inc.exact_mode = exact;
  {
    std::ifstream in(base_csv);
    if (!in.is_open()) {
      std::cerr << "cannot open " << base_csv << "\n";
      return 2;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      auto f = split_csv(line);
      if (f.size() < 2)
        continue;
      try {
        inc.insert(std::stod(f[0]), std::stod(f[1]));
      } catch (...) {
      }
    }
  }

  std::ifstream in(stream_csv);
  if (!in.is_open()) {
    std::cerr << "cannot open " << stream_csv << "\n";
    return 2;
  }
  std::ofstream out(out_csv);
  out << "step,operation,incr_ns,full_ns,n_active,incr_ari\n";

  std::string line;
  long long step = 0;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto f = split_csv(line);
    if (f.empty())
      continue;
    const std::string op = f[0];

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (op == "insert" && f.size() >= 3)
      inc.insert(std::stod(f[1]), std::stod(f[2]));
    else if (op == "delete" && f.size() >= 3)
      inc.remove_nearest(std::stod(f[1]), std::stod(f[2]));
    else if (op == "move" && f.size() >= 5)
      inc.move_nearest(std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
                       std::stod(f[4]));
    else
      continue;
    const auto t1 = std::chrono::high_resolution_clock::now();
    const long long incr_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    ++step;

    long long full_ns = -1;
    double ari = -2.0;
    if (step % every == 0) {
      // Ground truth: from-scratch DBSCAN over the currently active points.
      std::vector<Pt> act;
      std::vector<std::size_t> act_idx;
      act.reserve(inc.n_active);
      for (std::size_t i = 0; i < inc.pts.size(); ++i)
        if (inc.active[i]) {
          act.push_back(inc.pts[i]);
          act_idx.push_back(i);
        }
      const auto f0 = std::chrono::high_resolution_clock::now();
      const std::vector<int> gt = full_dbscan(act, eps, min_pts);
      const auto f1 = std::chrono::high_resolution_clock::now();
      full_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(f1 - f0).count();
      // Map ground-truth labels (over compacted active indices) back to full ids.
      const std::vector<int> inc_lab = inc.labels();
      std::vector<int> gt_full(inc.pts.size(), -2);
      for (std::size_t j = 0; j < act_idx.size(); ++j)
        gt_full[act_idx[j]] = gt[j];
      ari = adjusted_rand_index(inc_lab, gt_full, act_idx);
    }
    out << step << "," << op << "," << incr_ns << "," << full_ns << ","
        << inc.n_active << "," << ari << "\n";
  }
  out.close();
  return 0;
}
