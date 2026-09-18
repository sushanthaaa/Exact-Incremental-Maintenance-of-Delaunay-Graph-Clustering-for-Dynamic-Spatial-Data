#include "DelaunayClusterer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace delaucluster {
namespace {

constexpr double kPaddingFraction = 0.01;
constexpr double kMinPadding = 1e-9;
constexpr std::size_t kMaxConnectivitySummaryClusterSize = 256;
constexpr std::size_t kMaxSmallSideSplitSearch = 512;

struct EdgeKey {
  std::size_t a;
  std::size_t b;
  bool operator==(const EdgeKey &other) const {
    return a == other.a && b == other.b;
  }
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &key) const {
    return key.a * 1315423911u + key.b;
  }
};

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> size;

  explicit DisjointSet(std::size_t n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (size[a] < size[b])
      std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
    return true;
  }
};

struct ThresholdEvaluation {
  double threshold = 0.0;
  std::size_t num_clusters = 0;
  std::size_t noise_count = 0;
  std::size_t retained_edges = 0;
  std::size_t largest_cluster = 0;
  double component_entropy = 0.0;
  double base_objective = -std::numeric_limits<double>::infinity();
  double stability_adjustment = 0.0;
  std::size_t plateau_width_candidates = 1;
  double plateau_center_offset = 0.0;
  double largest_jump_before_pct = 0.0;
  double largest_jump_after_pct = 0.0;
  double noise_drop_after_pct = 0.0;
  double objective = -std::numeric_limits<double>::infinity();
};

double sqr(double v) { return v * v; }

double distance_xy(double ax, double ay, double bx, double by) {
  return std::sqrt(sqr(ax - bx) + sqr(ay - by));
}

std::size_t desired_bucket_dimension(std::size_t active_count) {
  if (active_count == 0)
    return 0;
  return std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(std::sqrt(static_cast<double>(active_count)))));
}

template <typename T> void erase_value(std::vector<T> &values, const T &value) {
  values.erase(std::remove(values.begin(), values.end(), value), values.end());
}

template <typename T> std::vector<T> sorted_copy(std::vector<T> values) {
  std::sort(values.begin(), values.end());
  return values;
}

std::pair<std::size_t, std::size_t> ordered_pair(std::size_t a,
                                                 std::size_t b) {
  if (a > b)
    std::swap(a, b);
  return {a, b};
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::string normalized_threshold_mode(const ClusterOptions &options) {
  if (options.global_threshold_ablation)
    return "raw_length";
  std::string mode = lower_copy(options.threshold_mode);
  if (mode == "local" || mode == "normalized")
    mode = "local_scale";
  if (mode == "raw" || mode == "global")
    mode = "raw_length";
  if (mode == "bucket" || mode == "bucket_percentile")
    mode = "bucket_percentile";
  if (mode == "knn" || mode == "k_neighborhood" || mode == "k_neighbourhood")
    mode = "knn_median";
  if (mode == "bridge_guard" || mode == "supported" ||
      mode == "supported_merge")
    mode = "supported_merge";
  if (mode == "saddle" || mode == "saddle_cut")
    mode = "saddle_cut";
  if (mode == "manifold" || mode == "manifold_filter" ||
      mode == "manifold_continuity")
    mode = "manifold_filter";
  if (mode != "raw_length" && mode != "local_scale" &&
      mode != "bucket_percentile" && mode != "knn_median" &&
      mode != "supported_merge" && mode != "saddle_cut" &&
      mode != "manifold_filter") {
    return "local_scale";
  }
  return mode;
}

bool threshold_mode_uses_local_scale(const std::string &mode) {
  return mode != "raw_length";
}

bool threshold_mode_supports_local_edge_refresh(const ClusterOptions &options) {
  const std::string mode = normalized_threshold_mode(options);
  return mode == "raw_length" || mode == "local_scale";
}

bool connectivity_summary_enabled(const ClusterOptions &options) {
  return options.experimental_local_relabel &&
         options.experimental_local_edge_refresh &&
         options.experimental_connectivity_summary;
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> parts;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    parts.push_back(item);
  }
  return parts;
}

bool parse_double(const std::string &s, double &out) {
  try {
    size_t used = 0;
    out = std::stod(s, &used);
    return used == s.size();
  } catch (...) {
    return false;
  }
}

// Bounded local connectivity probes. These take an on-demand neighbor getter
// `nbrs(id) -> vector<id>` (the retained, active neighbors of id) instead of a
// fully-materialized adjacency map, so a dynamic update never builds an O(V+E)
// graph: each probe explores only the local region it needs.
template <typename NbrFn>
bool reachable_in_adjacency(const NbrFn &nbrs, std::size_t start,
                            std::size_t target) {
  if (start == target)
    return true;
  std::unordered_set<std::size_t> visited;
  std::queue<std::size_t> q;
  visited.insert(start);
  q.push(start);
  while (!q.empty()) {
    const std::size_t curr = q.front();
    q.pop();
    for (std::size_t nb : nbrs(curr)) {
      if (nb == target)
        return true;
      if (visited.insert(nb).second)
        q.push(nb);
    }
  }
  return false;
}

template <typename NbrFn>
bool reachable_in_adjacency_without_edge(const NbrFn &nbrs, std::size_t start,
                                         std::size_t target) {
  if (start == target)
    return true;
  std::unordered_set<std::size_t> visited;
  std::queue<std::size_t> q;
  visited.insert(start);
  q.push(start);
  while (!q.empty()) {
    const std::size_t curr = q.front();
    q.pop();
    for (std::size_t nb : nbrs(curr)) {
      if ((curr == start && nb == target) ||
          (curr == target && nb == start)) {
        continue;
      }
      if (nb == target)
        return true;
      if (visited.insert(nb).second)
        q.push(nb);
    }
  }
  return false;
}

template <typename NbrFn>
bool local_path_witness(const NbrFn &nbrs, std::size_t start, std::size_t target,
                        std::size_t max_visits, std::size_t *visited_count) {
  if (visited_count)
    *visited_count = 0;
  if (start == target)
    return true;
  const std::vector<std::size_t> start_nbrs = nbrs(start);
  const std::vector<std::size_t> target_nbrs = nbrs(target);
  std::unordered_set<std::size_t> target_neighbors(target_nbrs.begin(),
                                                   target_nbrs.end());
  for (std::size_t nb : start_nbrs) {
    if (nb == target || target_neighbors.find(nb) != target_neighbors.end()) {
      if (visited_count)
        *visited_count = 3;
      return true;
    }
  }

  std::unordered_set<std::size_t> visited;
  std::queue<std::size_t> q;
  visited.insert(start);
  q.push(start);
  while (!q.empty()) {
    const std::size_t curr = q.front();
    q.pop();
    if (visited_count)
      *visited_count = std::max(*visited_count, visited.size());
    if (visited.size() > max_visits)
      return false;

    for (std::size_t nb : nbrs(curr)) {
      if (nb == target) {
        if (visited_count)
          *visited_count = std::max(*visited_count, visited.size());
        return true;
      }
      if (visited.insert(nb).second)
        q.push(nb);
      if (visited.size() > max_visits)
        return false;
    }
  }
  return false;
}

void bridge_articulation_summary(
    const std::unordered_map<std::size_t, std::vector<std::size_t>> &adj,
    std::set<std::pair<std::size_t, std::size_t>> &bridges,
    std::set<std::size_t> &articulations) {
  std::unordered_map<std::size_t, int> discovery;
  std::unordered_map<std::size_t, int> low;
  std::unordered_map<std::size_t, std::size_t> parent;
  int timestamp = 0;

  std::function<void(std::size_t)> dfs = [&](std::size_t u) {
    discovery[u] = low[u] = ++timestamp;
    int child_count = 0;
    bool articulation = false;

    const auto adj_it = adj.find(u);
    if (adj_it != adj.end()) {
      for (std::size_t v : adj_it->second) {
        if (discovery.find(v) == discovery.end()) {
          parent[v] = u;
          child_count++;
          dfs(v);
          low[u] = std::min(low[u], low[v]);
          if (low[v] > discovery[u])
            bridges.insert(ordered_pair(u, v));
          if (parent.find(u) != parent.end() && low[v] >= discovery[u])
            articulation = true;
        } else {
          auto parent_it = parent.find(u);
          if (parent_it == parent.end() || parent_it->second != v)
            low[u] = std::min(low[u], discovery[v]);
        }
      }
    }

    if (parent.find(u) == parent.end() && child_count > 1)
      articulation = true;
    if (articulation)
      articulations.insert(u);
  };

  for (const auto &kv : adj) {
    if (discovery.find(kv.first) == discovery.end())
      dfs(kv.first);
  }
}

double median_sorted(const std::vector<double> &sorted) {
  if (sorted.empty())
    return 0.0;
  const std::size_t n = sorted.size();
  if (n % 2 == 1)
    return sorted[n / 2];
  return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

double percentile_sorted(const std::vector<double> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  if (p <= 0)
    return sorted.front();
  if (p >= 1)
    return sorted.back();
  double pos = p * static_cast<double>(sorted.size() - 1);
  std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

double robust_threshold(std::vector<double> values, double iqr_multiplier) {
  if (values.empty())
    return std::numeric_limits<double>::infinity();

  std::sort(values.begin(), values.end());
  const double q1 = percentile_sorted(values, 0.25);
  const double q3 = percentile_sorted(values, 0.75);
  const double iqr = q3 - q1;
  double threshold = q3 + iqr_multiplier * iqr;

  if (iqr <= 1e-12) {
    const double med = median_sorted(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double v : values)
      deviations.push_back(std::abs(v - med));
    std::sort(deviations.begin(), deviations.end());
    const double mad_sigma = 1.4826 * median_sorted(deviations);
    threshold = med + iqr_multiplier * mad_sigma;
  }

  if (threshold <= 0 || !std::isfinite(threshold))
    threshold = values.back();
  return threshold;
}

ThresholdEvaluation evaluate_threshold_with_caps(
    const std::vector<PointRecord> &records, const std::vector<EdgeRecord> &edges,
    const std::vector<double> *edge_caps, double threshold,
    int min_cluster_size) {
  ThresholdEvaluation eval;
  eval.threshold = threshold;

  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  std::size_t active_count = 0;
  for (const auto &record : records) {
    if (record.active) {
      adj[record.id] = {};
      active_count++;
    }
  }
  if (active_count == 0)
    return eval;

  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const auto &edge = edges[edge_index];
    double edge_threshold = threshold;
    if (edge_caps != nullptr && edge_index < edge_caps->size()) {
      edge_threshold = std::min(edge_threshold, (*edge_caps)[edge_index]);
    }
    if (edge.normalized_score > edge_threshold)
      continue;
    auto u = adj.find(edge.u);
    auto v = adj.find(edge.v);
    if (u == adj.end() || v == adj.end())
      continue;
    u->second.push_back(edge.v);
    v->second.push_back(edge.u);
    eval.retained_edges++;
  }

  std::unordered_set<std::size_t> visited;
  std::vector<std::size_t> cluster_sizes;
  for (const auto &record : records) {
    if (!record.active || visited.find(record.id) != visited.end())
      continue;

    std::size_t component_size = 0;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component_size++;
      for (std::size_t nb : adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }

    if (static_cast<int>(component_size) < min_cluster_size) {
      eval.noise_count += component_size;
    } else {
      eval.num_clusters++;
      eval.largest_cluster = std::max(eval.largest_cluster, component_size);
      cluster_sizes.push_back(component_size);
    }
  }

  double entropy = 0.0;
  for (std::size_t size : cluster_sizes) {
    const double p =
        static_cast<double>(size) /
        static_cast<double>(std::max<std::size_t>(1, active_count));
    if (p > 0.0)
      entropy -= p * std::log(p);
  }
  const double max_entropy =
      eval.num_clusters > 1
          ? std::log(static_cast<double>(eval.num_clusters))
          : 1.0;
  eval.component_entropy =
      max_entropy > 0.0 ? std::min(1.0, entropy / max_entropy) : 0.0;

  const double active =
      static_cast<double>(std::max<std::size_t>(1, active_count));
  const double noise_fraction = static_cast<double>(eval.noise_count) / active;
  const double non_noise_fraction = 1.0 - noise_fraction;
  const double largest_fraction =
      static_cast<double>(eval.largest_cluster) / active;
  const double giant_excess = std::max(0.0, largest_fraction - 0.65);
  const double noise_excess = std::max(0.0, noise_fraction - 0.30);
  const double fragmentation =
      static_cast<double>(eval.num_clusters) /
      std::sqrt(active + static_cast<double>(eval.num_clusters));

  eval.objective =
      0.80 * non_noise_fraction + 0.35 * eval.component_entropy -
      0.45 * largest_fraction * largest_fraction -
      1.50 * giant_excess * giant_excess -
      0.75 * noise_excess * noise_excess -
      0.08 * std::log2(static_cast<double>(eval.num_clusters) + 1.0) -
      0.15 * fragmentation;
  if (eval.num_clusters < 2)
    eval.objective -= 1.50;
  eval.base_objective = eval.objective;

  return eval;
}

ThresholdEvaluation evaluate_threshold(
    const std::vector<PointRecord> &records, const std::vector<EdgeRecord> &edges,
    double threshold, int min_cluster_size) {
  return evaluate_threshold_with_caps(records, edges, nullptr, threshold,
                                      min_cluster_size);
}

ThresholdEvaluation evaluate_current_retained(
    const std::vector<PointRecord> &records,
    const std::vector<EdgeRecord> &edges, double threshold,
    int min_cluster_size) {
  ThresholdEvaluation eval;
  eval.threshold = threshold;

  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  std::size_t active_count = 0;
  for (const auto &record : records) {
    if (record.active) {
      adj[record.id] = {};
      active_count++;
    }
  }
  if (active_count == 0)
    return eval;

  for (const auto &edge : edges) {
    if (!edge.retained)
      continue;
    auto u = adj.find(edge.u);
    auto v = adj.find(edge.v);
    if (u == adj.end() || v == adj.end())
      continue;
    u->second.push_back(edge.v);
    v->second.push_back(edge.u);
    eval.retained_edges++;
  }

  std::unordered_set<std::size_t> visited;
  std::vector<std::size_t> cluster_sizes;
  for (const auto &record : records) {
    if (!record.active || visited.find(record.id) != visited.end())
      continue;

    std::size_t component_size = 0;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component_size++;
      for (std::size_t nb : adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }

    if (static_cast<int>(component_size) < min_cluster_size) {
      eval.noise_count += component_size;
    } else {
      eval.num_clusters++;
      eval.largest_cluster = std::max(eval.largest_cluster, component_size);
      cluster_sizes.push_back(component_size);
    }
  }

  double entropy = 0.0;
  for (std::size_t size : cluster_sizes) {
    const double p =
        static_cast<double>(size) /
        static_cast<double>(std::max<std::size_t>(1, active_count));
    if (p > 0.0)
      entropy -= p * std::log(p);
  }
  const double max_entropy =
      eval.num_clusters > 1
          ? std::log(static_cast<double>(eval.num_clusters))
          : 1.0;
  eval.component_entropy =
      max_entropy > 0.0 ? std::min(1.0, entropy / max_entropy) : 0.0;

  const double active =
      static_cast<double>(std::max<std::size_t>(1, active_count));
  const double noise_fraction = static_cast<double>(eval.noise_count) / active;
  const double non_noise_fraction = 1.0 - noise_fraction;
  const double largest_fraction =
      static_cast<double>(eval.largest_cluster) / active;
  const double giant_excess = std::max(0.0, largest_fraction - 0.65);
  const double noise_excess = std::max(0.0, noise_fraction - 0.30);
  const double fragmentation =
      static_cast<double>(eval.num_clusters) /
      std::sqrt(active + static_cast<double>(eval.num_clusters));

  eval.objective =
      0.80 * non_noise_fraction + 0.35 * eval.component_entropy -
      0.45 * largest_fraction * largest_fraction -
      1.50 * giant_excess * giant_excess -
      0.75 * noise_excess * noise_excess -
      0.08 * std::log2(static_cast<double>(eval.num_clusters) + 1.0) -
      0.15 * fragmentation;
  if (eval.num_clusters < 2)
    eval.objective -= 1.50;
  eval.base_objective = eval.objective;
  return eval;
}

std::size_t count_active_records(const std::vector<PointRecord> &records) {
  std::size_t active_count = 0;
  for (const auto &record : records) {
    if (record.active)
      active_count++;
  }
  return active_count;
}

void apply_stability_objective_adjustment(
    std::vector<ThresholdEvaluation> &evaluated, std::size_t active_count,
    std::size_t edge_count) {
  if (evaluated.empty())
    return;

  const double active =
      static_cast<double>(std::max<std::size_t>(1, active_count));
  const double edges =
      static_cast<double>(std::max<std::size_t>(1, edge_count));
  const double scan_denominator =
      static_cast<double>(std::max<std::size_t>(1, evaluated.size() - 1));

  for (std::size_t i = 0; i < evaluated.size(); ++i) {
    auto &eval = evaluated[i];
    std::size_t plateau_start = i;
    std::size_t plateau_end = i;
    while (plateau_start > 0 &&
           evaluated[plateau_start - 1].num_clusters == eval.num_clusters) {
      --plateau_start;
    }
    while (plateau_end + 1 < evaluated.size() &&
           evaluated[plateau_end + 1].num_clusters == eval.num_clusters) {
      ++plateau_end;
    }

    const double plateau_width =
        static_cast<double>(plateau_end - plateau_start + 1);
    const double plateau_center =
        0.5 * static_cast<double>(plateau_start + plateau_end);
    const double plateau_radius = std::max(1.0, 0.5 * (plateau_width - 1.0));
    const double plateau_center_offset =
        static_cast<double>(i) - plateau_center;
    const double center_bonus =
        std::max(0.0, 1.0 - std::abs(plateau_center_offset) / plateau_radius);

    const double largest_before =
        i == 0 ? 0.0
               : std::max(0.0,
                          static_cast<double>(eval.largest_cluster) -
                              static_cast<double>(
                                  evaluated[i - 1].largest_cluster)) /
                     active;
    const double largest_after =
        i + 1 >= evaluated.size()
            ? 0.0
            : std::max(0.0,
                       static_cast<double>(evaluated[i + 1].largest_cluster) -
                           static_cast<double>(eval.largest_cluster)) /
                  active;
    const double recent_largest_jump =
        std::max(largest_before,
                 i > 1
                     ? std::max(0.0,
                                static_cast<double>(
                                    evaluated[i - 1].largest_cluster) -
                                    static_cast<double>(
                                        evaluated[i - 2].largest_cluster)) /
                           active
                     : 0.0);
    const double noise_drop_after =
        i + 1 >= evaluated.size()
            ? 0.0
            : std::max(0.0,
                       static_cast<double>(eval.noise_count) -
                           static_cast<double>(evaluated[i + 1].noise_count)) /
                  active;
    const double retained_fraction =
        static_cast<double>(eval.retained_edges) / edges;
    const double scan_position = static_cast<double>(i) / scan_denominator;
    const double late_excess = std::max(0.0, scan_position - 0.85);
    const double retained_excess = std::max(0.0, retained_fraction - 0.70);
    const double plateau_bonus =
        0.04 * (plateau_width - 1.0) / scan_denominator;
    const double center_adjustment = 0.010 * center_bonus;
    const double jump_penalty = 0.12 * recent_largest_jump;
    const double late_penalty = 0.05 * late_excess * late_excess;
    const double retained_penalty = 0.12 * retained_excess * retained_excess;
    const double noise_knee_bonus = 0.015 * std::min(noise_drop_after, 0.20);

    eval.plateau_width_candidates = plateau_end - plateau_start + 1;
    eval.plateau_center_offset = plateau_center_offset;
    eval.largest_jump_before_pct = 100.0 * largest_before;
    eval.largest_jump_after_pct = 100.0 * largest_after;
    eval.noise_drop_after_pct = 100.0 * noise_drop_after;
    eval.stability_adjustment =
        plateau_bonus + center_adjustment + noise_knee_bonus - jump_penalty -
        late_penalty - retained_penalty;
    eval.objective = eval.base_objective + eval.stability_adjustment;
  }
}

ThresholdTraceRecord make_threshold_trace_record(
    const ThresholdEvaluation &eval, std::size_t candidate_index, bool selected,
    bool nontrivial, std::size_t active_count, std::size_t edge_count) {
  const double active =
      static_cast<double>(std::max<std::size_t>(1, active_count));
  ThresholdTraceRecord row;
  row.candidate_index = candidate_index;
  row.threshold = eval.threshold;
  row.selected = selected;
  row.nontrivial = nontrivial;
  row.active_points = active_count;
  row.edge_count = edge_count;
  row.retained_edges = eval.retained_edges;
  row.retained_edge_pct =
      100.0 * static_cast<double>(eval.retained_edges) /
      static_cast<double>(std::max<std::size_t>(1, edge_count));
  row.num_clusters = eval.num_clusters;
  row.noise_count = eval.noise_count;
  row.noise_pct = 100.0 * static_cast<double>(eval.noise_count) / active;
  row.largest_cluster = eval.largest_cluster;
  row.largest_cluster_pct =
      100.0 * static_cast<double>(eval.largest_cluster) / active;
  row.component_entropy = eval.component_entropy;
  row.base_objective = eval.base_objective;
  row.stability_adjustment = eval.stability_adjustment;
  row.plateau_width_candidates = eval.plateau_width_candidates;
  row.plateau_center_offset = eval.plateau_center_offset;
  row.largest_jump_before_pct = eval.largest_jump_before_pct;
  row.largest_jump_after_pct = eval.largest_jump_after_pct;
  row.noise_drop_after_pct = eval.noise_drop_after_pct;
  row.objective = eval.objective;
  return row;
}

std::unordered_map<std::size_t, std::string>
canonical_partition(const std::vector<PointRecord> &records) {
  std::unordered_map<int, std::vector<std::size_t>> by_cluster;
  std::unordered_map<std::size_t, std::string> signature;
  for (const auto &record : records) {
    if (!record.active)
      continue;
    if (record.is_noise || record.cluster_id < 0) {
      signature[record.id] = "noise";
    } else {
      by_cluster[record.cluster_id].push_back(record.id);
    }
  }

  for (auto &kv : by_cluster) {
    auto &ids = kv.second;
    std::sort(ids.begin(), ids.end());
    std::ostringstream joined;
    joined << "cluster";
    for (std::size_t id : ids)
      joined << ":" << id;
    const std::string value = joined.str();
    for (std::size_t id : ids)
      signature[id] = value;
  }

  return signature;
}

std::size_t partition_mismatches(const std::vector<PointRecord> &left,
                                 const std::vector<PointRecord> &right) {
  const auto l_sig = canonical_partition(left);
  const auto r_sig = canonical_partition(right);
  std::unordered_set<std::size_t> ids;
  for (const auto &kv : l_sig)
    ids.insert(kv.first);
  for (const auto &kv : r_sig)
    ids.insert(kv.first);

  std::size_t mismatches = 0;
  for (std::size_t id : ids) {
    auto l = l_sig.find(id);
    auto r = r_sig.find(id);
    if (l == l_sig.end() || r == r_sig.end() || l->second != r->second)
      mismatches++;
  }
  return mismatches;
}

void ensure_parent_dir(const std::string &path) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
}

} // namespace

void SpatialBucketIndex::clear() {
  rows_ = cols_ = 0;
  min_x_ = max_x_ = min_y_ = max_y_ = 0.0;
  step_x_ = step_y_ = 1.0;
  cells_.clear();
  point_to_cell_.clear();
  edge_to_cells_.clear();
}

bool SpatialBucketIndex::contains_point(double x, double y) const {
  if (rows_ == 0 || cols_ == 0)
    return false;
  const double eps_x = std::max(step_x_ * 1e-12, kMinPadding);
  const double eps_y = std::max(step_y_ * 1e-12, kMinPadding);
  return x >= min_x_ - eps_x && x <= max_x_ + eps_x &&
         y >= min_y_ - eps_y && y <= max_y_ + eps_y;
}

int SpatialBucketIndex::bucket_index(double x, double y) const {
  if (rows_ == 0 || cols_ == 0)
    return -1;
  int col = static_cast<int>((x - min_x_) / step_x_);
  int row = static_cast<int>((y - min_y_) / step_y_);
  col = std::max(0, std::min(col, static_cast<int>(cols_) - 1));
  row = std::max(0, std::min(row, static_cast<int>(rows_) - 1));
  return row * static_cast<int>(cols_) + col;
}

int SpatialBucketIndex::cell_row(int cell_index) const {
  if (cols_ == 0 || cell_index < 0)
    return -1;
  return cell_index / static_cast<int>(cols_);
}

int SpatialBucketIndex::cell_col(int cell_index) const {
  if (cols_ == 0 || cell_index < 0)
    return -1;
  return cell_index % static_cast<int>(cols_);
}

std::vector<int> SpatialBucketIndex::cell_neighborhood(int cell_index,
                                                       int radius) const {
  std::vector<int> cells;
  if (rows_ == 0 || cols_ == 0 || cell_index < 0)
    return cells;
  const int row = cell_row(cell_index);
  const int col = cell_col(cell_index);
  if (row < 0 || col < 0 || row >= static_cast<int>(rows_) ||
      col >= static_cast<int>(cols_))
    return cells;
  const int r = std::max(0, radius);
  for (int rr = row - r; rr <= row + r; ++rr) {
    if (rr < 0 || rr >= static_cast<int>(rows_))
      continue;
    for (int cc = col - r; cc <= col + r; ++cc) {
      if (cc < 0 || cc >= static_cast<int>(cols_))
        continue;
      cells.push_back(rr * static_cast<int>(cols_) + cc);
    }
  }
  return cells;
}

std::vector<int> SpatialBucketIndex::segment_bucket_indices(double x0,
                                                            double y0,
                                                            double x1,
                                                            double y1) const {
  std::vector<int> traversed;
  if (rows_ == 0 || cols_ == 0)
    return traversed;

  const int start = bucket_index(x0, y0);
  const int finish = bucket_index(x1, y1);
  if (start < 0 || finish < 0)
    return traversed;

  int col = start % static_cast<int>(cols_);
  int row = start / static_cast<int>(cols_);
  const int target_col = finish % static_cast<int>(cols_);
  const int target_row = finish / static_cast<int>(cols_);

  auto append = [&traversed, this](int r, int c) {
    if (r < 0 || c < 0 || r >= static_cast<int>(rows_) ||
        c >= static_cast<int>(cols_))
      return;
    const int idx = r * static_cast<int>(cols_) + c;
    if (std::find(traversed.begin(), traversed.end(), idx) ==
        traversed.end()) {
      traversed.push_back(idx);
    }
  };

  append(row, col);
  if (start == finish)
    return traversed;

  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const int step_col = (dx > 0.0) ? 1 : (dx < 0.0 ? -1 : 0);
  const int step_row = (dy > 0.0) ? 1 : (dy < 0.0 ? -1 : 0);
  double t_max_x = std::numeric_limits<double>::infinity();
  double t_max_y = std::numeric_limits<double>::infinity();
  double t_delta_x = std::numeric_limits<double>::infinity();
  double t_delta_y = std::numeric_limits<double>::infinity();

  if (step_col != 0) {
    const double boundary =
        min_x_ + static_cast<double>(step_col > 0 ? col + 1 : col) * step_x_;
    t_max_x = (boundary - x0) / dx;
    t_delta_x = step_x_ / std::abs(dx);
  }
  if (step_row != 0) {
    const double boundary =
        min_y_ + static_cast<double>(step_row > 0 ? row + 1 : row) * step_y_;
    t_max_y = (boundary - y0) / dy;
    t_delta_y = step_y_ / std::abs(dy);
  }

  const std::size_t guard = rows_ + cols_ + 4;
  for (std::size_t steps = 0;
       steps < guard && (col != target_col || row != target_row); ++steps) {
    if (t_max_x < t_max_y) {
      col += step_col;
      t_max_x += t_delta_x;
    } else if (t_max_y < t_max_x) {
      row += step_row;
      t_max_y += t_delta_y;
    } else {
      col += step_col;
      row += step_row;
      t_max_x += t_delta_x;
      t_max_y += t_delta_y;
    }
    append(row, col);
  }

  return traversed;
}

void SpatialBucketIndex::remove_point(std::size_t point_id) {
  auto it = point_to_cell_.find(point_id);
  if (it == point_to_cell_.end())
    return;
  const int idx = it->second;
  if (idx >= 0 && static_cast<std::size_t>(idx) < cells_.size())
    erase_value(cells_[static_cast<std::size_t>(idx)].point_ids, point_id);
  point_to_cell_.erase(it);
}

void SpatialBucketIndex::upsert_point(std::size_t point_id, double x,
                                      double y) {
  remove_point(point_id);
  const int idx = bucket_index(x, y);
  if (idx < 0 || static_cast<std::size_t>(idx) >= cells_.size())
    return;
  auto &ids = cells_[static_cast<std::size_t>(idx)].point_ids;
  if (std::find(ids.begin(), ids.end(), point_id) == ids.end())
    ids.push_back(point_id);
  point_to_cell_[point_id] = idx;
}

void SpatialBucketIndex::remove_edge(std::size_t edge_id) {
  auto it = edge_to_cells_.find(edge_id);
  if (it == edge_to_cells_.end())
    return;
  for (int idx : it->second) {
    if (idx >= 0 && static_cast<std::size_t>(idx) < cells_.size())
      erase_value(cells_[static_cast<std::size_t>(idx)].edge_ids, edge_id);
  }
  edge_to_cells_.erase(it);
}

void SpatialBucketIndex::upsert_edge(std::size_t edge_id, double x0, double y0,
                                     double x1, double y1) {
  remove_edge(edge_id);
  auto traversed = segment_bucket_indices(x0, y0, x1, y1);
  if (traversed.empty())
    return;
  edge_to_cells_[edge_id] = traversed;
  for (int idx : traversed) {
    if (idx >= 0 && static_cast<std::size_t>(idx) < cells_.size())
      cells_[static_cast<std::size_t>(idx)].edge_ids.push_back(edge_id);
  }
}

void SpatialBucketIndex::rebuild_on_current_layout(
    const std::vector<PointRecord> &points,
    const std::vector<EdgeRecord> &edges) {
  if (rows_ == 0 || cols_ == 0) {
    clear();
    return;
  }

  cells_.assign(rows_ * cols_, BucketCell{});
  point_to_cell_.clear();
  edge_to_cells_.clear();

  std::unordered_map<std::size_t, const PointRecord *> by_id;
  for (const auto &point : points) {
    if (!point.active)
      continue;
    by_id[point.id] = &point;
    upsert_point(point.id, point.x, point.y);
  }

  for (const auto &edge : edges) {
    auto u = by_id.find(edge.u);
    auto v = by_id.find(edge.v);
    if (u == by_id.end() || v == by_id.end())
      continue;
    upsert_edge(edge.id, u->second->x, u->second->y, v->second->x,
                v->second->y);
  }
}

std::size_t SpatialBucketIndex::nonempty_point_cells() const {
  return static_cast<std::size_t>(std::count_if(
      cells_.begin(), cells_.end(),
      [](const BucketCell &cell) { return !cell.point_ids.empty(); }));
}

std::size_t SpatialBucketIndex::nonempty_edge_cells() const {
  return static_cast<std::size_t>(std::count_if(
      cells_.begin(), cells_.end(),
      [](const BucketCell &cell) { return !cell.edge_ids.empty(); }));
}

std::size_t SpatialBucketIndex::max_points_per_cell() const {
  std::size_t best = 0;
  for (const auto &cell : cells_)
    best = std::max(best, cell.point_ids.size());
  return best;
}

std::size_t SpatialBucketIndex::max_edges_per_cell() const {
  std::size_t best = 0;
  for (const auto &cell : cells_)
    best = std::max(best, cell.edge_ids.size());
  return best;
}

double SpatialBucketIndex::mean_points_per_nonempty_cell() const {
  std::size_t total = 0;
  std::size_t nonempty = 0;
  for (const auto &cell : cells_) {
    if (!cell.point_ids.empty()) {
      total += cell.point_ids.size();
      nonempty++;
    }
  }
  return nonempty == 0 ? 0.0 : static_cast<double>(total) /
                                  static_cast<double>(nonempty);
}

double SpatialBucketIndex::mean_edges_per_nonempty_cell() const {
  std::size_t total = 0;
  std::size_t nonempty = 0;
  for (const auto &cell : cells_) {
    if (!cell.edge_ids.empty()) {
      total += cell.edge_ids.size();
      nonempty++;
    }
  }
  return nonempty == 0 ? 0.0 : static_cast<double>(total) /
                                  static_cast<double>(nonempty);
}

void SpatialBucketIndex::build(const std::vector<PointRecord> &points,
                               const std::vector<EdgeRecord> &edges) {
  clear();
  std::vector<const PointRecord *> active;
  for (const auto &p : points) {
    if (p.active)
      active.push_back(&p);
  }
  if (active.empty())
    return;

  double min_x = active.front()->x, max_x = active.front()->x;
  double min_y = active.front()->y, max_y = active.front()->y;
  for (const auto *p : active) {
    min_x = std::min(min_x, p->x);
    max_x = std::max(max_x, p->x);
    min_y = std::min(min_y, p->y);
    max_y = std::max(max_y, p->y);
  }

  const double range_x = std::max(max_x - min_x, kMinPadding);
  const double range_y = std::max(max_y - min_y, kMinPadding);
  const double pad_x = std::max(range_x * kPaddingFraction, kMinPadding);
  const double pad_y = std::max(range_y * kPaddingFraction, kMinPadding);

  rows_ = cols_ = desired_bucket_dimension(active.size());
  min_x_ = min_x - pad_x;
  max_x_ = max_x + pad_x;
  min_y_ = min_y - pad_y;
  max_y_ = max_y + pad_y;
  step_x_ = std::max((max_x_ - min_x_) / static_cast<double>(cols_), kMinPadding);
  step_y_ = std::max((max_y_ - min_y_) / static_cast<double>(rows_), kMinPadding);
  cells_.resize(rows_ * cols_);

  for (const auto *p : active) {
    const int idx = bucket_index(p->x, p->y);
    if (idx >= 0) {
      cells_[static_cast<std::size_t>(idx)].point_ids.push_back(p->id);
      point_to_cell_[p->id] = idx;
    }
  }

  std::unordered_map<std::size_t, const PointRecord *> by_id;
  for (const auto *p : active)
    by_id[p->id] = p;

  for (const auto &edge : edges) {
    auto it_u = by_id.find(edge.u);
    auto it_v = by_id.find(edge.v);
    if (it_u == by_id.end() || it_v == by_id.end())
      continue;

    auto traversed =
        segment_bucket_indices(it_u->second->x, it_u->second->y,
                               it_v->second->x, it_v->second->y);
    edge_to_cells_[edge.id] = traversed;
    for (int idx : traversed) {
      cells_[static_cast<std::size_t>(idx)].edge_ids.push_back(edge.id);
    }
  }
}

std::vector<PointRecord> DelaunayClusterer::load_csv(
    const std::string &csv_path) {
  std::ifstream in(csv_path);
  if (!in.is_open())
    throw std::runtime_error("Cannot open CSV: " + csv_path);

  std::vector<PointRecord> points;
  std::string line;
  std::size_t id = 0;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto parts = split_csv_line(line);
    if (parts.size() < 2)
      continue;

    double x = 0.0, y = 0.0;
    if (!parse_double(parts[0], x) || !parse_double(parts[1], y)) {
      if (points.empty())
        continue; // tolerate a single header row
      throw std::runtime_error("Malformed numeric row in CSV: " + line);
    }

    PointRecord rec;
    rec.id = id++;
    rec.x = x;
    rec.y = y;
    if (parts.size() >= 3) {
      double label_value = 0.0;
      if (parse_double(parts[2], label_value)) {
        rec.label = static_cast<int>(std::llround(label_value));
        rec.has_label = true;
      }
    }
    points.push_back(rec);
  }
  return points;
}

void DelaunayClusterer::clear() {
  dt_.clear();
  records_.clear();
  id_to_index_.clear();
  vertex_handles_.clear();
  edges_.clear();
  edge_index_by_id_.clear();
  edge_ids_by_vertex_.clear();
  threshold_trace_.clear();
  last_local_edge_refresh_tracked_ = false;
  last_bucket_removed_edge_ids_.clear();
  last_bucket_upsert_edge_ids_.clear();
  last_removed_retained_edges_.clear();
  last_added_retained_edges_.clear();
  local_scale_.clear();
  bucket_index_.clear();
  stats_ = ClusterStats{};
  cluster_sizes_.clear();
  cluster_members_.clear();
  retained_bridge_edges_.clear();
  retained_articulation_vertices_.clear();
  retained_bridge_edges_by_cluster_.clear();
  retained_articulation_vertices_by_cluster_.clear();
  retained_connectivity_summary_clusters_.clear();
  last_relabel_removed_retained_edges_count_ = 0;
  last_relabel_added_retained_edges_count_ = 0;
  last_small_side_split_used_ = false;
  last_point_delete_split_used_ = false;
  next_id_ = 0;
  next_edge_id_ = 0;
}

void DelaunayClusterer::fit(const std::vector<PointRecord> &points) {
  clear();
  records_ = points;
  std::size_t max_id = 0;
  for (std::size_t i = 0; i < records_.size(); ++i) {
    records_[i].active = true;
    records_[i].cluster_id = -1;
    records_[i].is_noise = false;
    id_to_index_[records_[i].id] = i;
    max_id = std::max(max_id, records_[i].id);
  }
  next_id_ = records_.empty() ? 0 : max_id + 1;
  rebuild_model(nullptr);
}

void DelaunayClusterer::fit_csv(const std::string &csv_path) {
  fit(load_csv(csv_path));
}

void DelaunayClusterer::rebuild_model(std::size_t *changed_labels) {
  rebuild_triangulation();
  refresh_model_from_current_triangulation(changed_labels);
}

void DelaunayClusterer::refresh_bucket_index_and_stats() {
  bucket_index_.build(records_, edges_);
  refresh_bucket_stats_only();
}

void DelaunayClusterer::refresh_bucket_stats_only() {
  stats_.bucket_rows = bucket_index_.rows();
  stats_.bucket_cols = bucket_index_.cols();
  stats_.bucket_nonempty_point_cells = bucket_index_.nonempty_point_cells();
  stats_.bucket_nonempty_edge_cells = bucket_index_.nonempty_edge_cells();
  stats_.bucket_max_points = bucket_index_.max_points_per_cell();
  stats_.bucket_max_edges = bucket_index_.max_edges_per_cell();
  stats_.bucket_mean_points_nonempty =
      bucket_index_.mean_points_per_nonempty_cell();
  stats_.bucket_mean_edges_nonempty =
      bucket_index_.mean_edges_per_nonempty_cell();
}

void DelaunayClusterer::refresh_model_from_current_triangulation(
    std::size_t *changed_labels, bool skip_bucket_refresh) {
  extract_edges_and_scales();
  choose_threshold_and_prune();
  label_connected_components(changed_labels);
  if (!skip_bucket_refresh)
    refresh_bucket_index_and_stats();
}

bool DelaunayClusterer::refresh_bucket_index_locally(
    const std::set<std::size_t> &touched_point_ids,
    DynamicUpdateReport &report) {
  if (!options_.experimental_local_bucket_refresh)
    return false;

  const std::size_t desired = desired_bucket_dimension(stats_.point_count);
  if (desired == 0) {
    bucket_index_.clear();
    refresh_bucket_stats_only();
    report.bucket_refresh_global = false;
    report.bucket_refresh_cells = 0;
    report.bucket_refresh_points = 0;
    report.bucket_refresh_edges = 0;
    return true;
  }

  if (bucket_index_.rows() != desired || bucket_index_.cols() != desired)
    return false;

  for (std::size_t id : touched_point_ids) {
    auto idx = id_to_index_.find(id);
    if (idx == id_to_index_.end())
      continue;
    const PointRecord &record = records_[idx->second];
    if (record.active && !bucket_index_.contains_point(record.x, record.y))
      return false;
  }

  if (!last_local_edge_refresh_tracked_)
    return false;

  std::set<int> refreshed_cells;
  auto add_cell = [&refreshed_cells](int cell) {
    if (cell >= 0)
      refreshed_cells.insert(cell);
  };
  auto add_cells = [&add_cell](const std::vector<int> &cells) {
    for (int cell : cells)
      add_cell(cell);
  };

  for (std::size_t id : touched_point_ids) {
    auto old_cell = bucket_index_.point_to_cell().find(id);
    if (old_cell != bucket_index_.point_to_cell().end())
      add_cell(old_cell->second);
    bucket_index_.remove_point(id);
    auto current_it = id_to_index_.find(id);
    if (current_it == id_to_index_.end())
      continue;
    const PointRecord &p = records_[current_it->second];
    if (!p.active)
      continue;
    add_cell(bucket_index_.bucket_index(p.x, p.y));
    bucket_index_.upsert_point(id, p.x, p.y);
  }

  for (std::size_t id : last_bucket_removed_edge_ids_) {
    auto old_cells = bucket_index_.edge_to_cells().find(id);
    if (old_cells != bucket_index_.edge_to_cells().end())
      add_cells(old_cells->second);
    bucket_index_.remove_edge(id);
  }

  for (std::size_t id : last_bucket_upsert_edge_ids_) {
    auto edge_it = edge_index_by_id_.find(id);
    if (edge_it == edge_index_by_id_.end())
      continue;
    const EdgeRecord &edge = edges_[edge_it->second];
    auto u = id_to_index_.find(edge.u);
    auto v = id_to_index_.find(edge.v);
    if (u == id_to_index_.end() || v == id_to_index_.end())
      continue;
    const PointRecord &pu = records_[u->second];
    const PointRecord &pv = records_[v->second];
    if (!pu.active || !pv.active)
      continue;
    const auto traversed =
        bucket_index_.segment_bucket_indices(pu.x, pu.y, pv.x, pv.y);
    add_cells(traversed);
    bucket_index_.upsert_edge(edge.id, pu.x, pu.y, pv.x, pv.y);
  }

  std::set<std::size_t> refreshed_edges = last_bucket_removed_edge_ids_;
  refreshed_edges.insert(last_bucket_upsert_edge_ids_.begin(),
                         last_bucket_upsert_edge_ids_.end());
  refresh_bucket_stats_only();
  report.bucket_refresh_global = false;
  report.bucket_refresh_cells = refreshed_cells.size();
  report.bucket_refresh_points = touched_point_ids.size();
  report.bucket_refresh_edges = refreshed_edges.size();
  return true;
}

void DelaunayClusterer::refresh_bucket_index_after_dynamic(
    const std::set<std::size_t> &touched_point_ids,
    DynamicUpdateReport &report) {
  report.local_bucket_refresh_used = options_.experimental_local_bucket_refresh;
  if (refresh_bucket_index_locally(touched_point_ids, report))
    return;

  refresh_bucket_index_and_stats();
  report.bucket_refresh_global = true;
  report.bucket_refresh_cells = bucket_index_.cells().size();
  report.bucket_refresh_points = stats_.point_count;
  report.bucket_refresh_edges = edges_.size();
}

void DelaunayClusterer::rebuild_model_with_local_relabel(
    const std::set<std::size_t> &seed_ids,
    const std::set<std::size_t> &active_added_ids_input,
    const std::set<std::size_t> &active_removed_ids_input,
    const std::unordered_map<std::size_t, std::vector<std::size_t>>
        &removed_old_retained_neighbors,
    std::size_t *changed_labels, std::size_t *relabel_scope_vertices,
    std::size_t *relabel_search_vertices,
    std::size_t *edge_refresh_scope_vertices,
    std::size_t *edge_refresh_candidate_edges, bool *edge_refresh_fallback,
    bool triangulation_is_current, bool skip_bucket_refresh) {
  if (!triangulation_is_current)
    rebuild_triangulation();
  last_relabel_removed_retained_edges_count_ = 0;
  last_relabel_added_retained_edges_count_ = 0;
  last_small_side_split_used_ = false;
  last_point_delete_split_used_ = false;

  bool local_edge_refresh_ok = false;
  if (options_.experimental_local_edge_refresh && triangulation_is_current) {
    local_edge_refresh_ok = refresh_edges_scales_and_pruning_locally(
        seed_ids, edge_refresh_scope_vertices, edge_refresh_candidate_edges);
  }
  if (edge_refresh_fallback)
    *edge_refresh_fallback =
        options_.experimental_local_edge_refresh && !local_edge_refresh_ok;
  if (!local_edge_refresh_ok) {
    extract_edges_and_scales();
    choose_threshold_and_prune();
    label_connected_components(changed_labels);
    if (relabel_scope_vertices)
      *relabel_scope_vertices = stats_.point_count;
    if (relabel_search_vertices)
      *relabel_search_vertices = stats_.point_count;
    if (!skip_bucket_refresh)
      refresh_bucket_index_and_stats();
    return;
  }

  // Ablation baseline only (--always-full-relabel): skip the heuristic local
  // relabel entirely and label by an unconditional full connected-components
  // pass over the exactly-refreshed retained graph. Exact by construction, so
  // the guard is redundant (and skips itself for this option).
  if (options_.always_full_relabel) {
    last_relabel_removed_retained_edges_count_ =
        last_removed_retained_edges_.size();
    last_relabel_added_retained_edges_count_ =
        last_added_retained_edges_.size();
    label_connected_components(changed_labels);
    if (relabel_scope_vertices)
      *relabel_scope_vertices = stats_.point_count;
    if (relabel_search_vertices)
      *relabel_search_vertices = stats_.point_count;
    if (!skip_bucket_refresh)
      refresh_bucket_index_and_stats();
    return;
  }

  // On-demand local adjacency: instead of materializing the full retained graph
  // every update (O(V+E)), query a vertex's retained, active neighbors lazily via
  // the incrementally-maintained edge registry. The relabel's connectivity probes
  // are all bounded/local, so this makes a typical update touch only the affected
  // region rather than the whole graph.
  auto is_active = [this](std::size_t id) {
    auto it = id_to_index_.find(id);
    return it != id_to_index_.end() && records_[it->second].active;
  };
  auto retained_neighbors = [this](std::size_t id) {
    std::vector<std::size_t> out;
    auto it = edge_ids_by_vertex_.find(id);
    if (it == edge_ids_by_vertex_.end())
      return out;
    for (std::size_t eid : it->second) {
      auto ei = edge_index_by_id_.find(eid);
      if (ei == edge_index_by_id_.end())
        continue;
      const EdgeRecord &e = edges_[ei->second];
      if (!e.retained)
        continue;
      const std::size_t other = (e.u == id) ? e.v : e.u;
      auto oi = id_to_index_.find(other);
      if (oi != id_to_index_.end() && records_[oi->second].active)
        out.push_back(other);
    }
    return out;
  };

  std::vector<std::pair<std::size_t, std::size_t>> removed_retained_edges;
  std::vector<std::pair<std::size_t, std::size_t>> added_retained_edges;
  removed_retained_edges.assign(last_removed_retained_edges_.begin(),
                                last_removed_retained_edges_.end());
  added_retained_edges.assign(last_added_retained_edges_.begin(),
                              last_added_retained_edges_.end());
  last_relabel_removed_retained_edges_count_ = removed_retained_edges.size();
  last_relabel_added_retained_edges_count_ = added_retained_edges.size();

  std::unordered_set<std::size_t> active_added_ids;
  std::unordered_set<std::size_t> active_removed_ids;
  for (std::size_t id : active_added_ids_input) {
    if (is_active(id))
      active_added_ids.insert(id);
  }
  for (std::size_t id : active_removed_ids_input) {
    if (!is_active(id))
      active_removed_ids.insert(id);
  }

  auto old_label = [this, &active_added_ids](std::size_t id, int &cluster,
                                             bool &is_noise) {
    if (active_added_ids.find(id) != active_added_ids.end())
      return false;
    auto idx = id_to_index_.find(id);
    if (idx == id_to_index_.end())
      return false;
    const PointRecord &record = records_[idx->second];
    cluster = record.cluster_id;
    is_noise = record.is_noise;
    return true;
  };

  auto same_old_non_noise_cluster = [&old_label](std::size_t a,
                                                 std::size_t b) {
    int cluster_a = -1;
    int cluster_b = -1;
    bool noise_a = true;
    bool noise_b = true;
    if (!old_label(a, cluster_a, noise_a) ||
        !old_label(b, cluster_b, noise_b)) {
      return false;
    }
    return !noise_a && !noise_b && cluster_a >= 0 && cluster_a == cluster_b;
  };

  auto old_noise = [&old_label](std::size_t id) {
    int cluster = -1;
    bool is_noise = true;
    if (!old_label(id, cluster, is_noise))
      return false;
    return is_noise || cluster < 0;
  };

  auto old_non_noise_cluster_id = [&old_label](std::size_t id, int &cluster) {
    bool is_noise = true;
    if (!old_label(id, cluster, is_noise))
      return false;
    return !is_noise && cluster >= 0;
  };

  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  bool summary_safe_removal_shape = active_added_ids.empty();
  std::size_t summary_non_noise_edge_removals = 0;
  std::size_t summary_non_noise_vertex_removals = 0;
  std::set<int> summary_affected_old_clusters;

  for (std::size_t id : active_removed_ids) {
    int cluster = -1;
    if (old_non_noise_cluster_id(id, cluster)) {
      summary_non_noise_vertex_removals++;
      summary_affected_old_clusters.insert(cluster);
    } else if (!old_noise(id)) {
      summary_safe_removal_shape = false;
    }
  }

  for (const auto &edge : removed_retained_edges) {
    if (active_removed_ids.find(edge.first) != active_removed_ids.end() ||
        active_removed_ids.find(edge.second) != active_removed_ids.end()) {
      continue;
    }
    int cluster_a = -1;
    int cluster_b = -1;
    if (old_non_noise_cluster_id(edge.first, cluster_a) &&
        old_non_noise_cluster_id(edge.second, cluster_b) &&
        cluster_a == cluster_b) {
      summary_non_noise_edge_removals++;
      summary_affected_old_clusters.insert(cluster_a);
    } else if (!(old_noise(edge.first) && old_noise(edge.second))) {
      summary_safe_removal_shape = false;
    }
  }

  const std::size_t summary_structural_removals =
      summary_non_noise_vertex_removals + summary_non_noise_edge_removals;
  const bool single_old_component_removal_summary =
      summary_safe_removal_shape && summary_structural_removals == 1 &&
      summary_affected_old_clusters.size() == 1;

  auto old_bridge_summary_certifies =
      [&](const std::pair<std::size_t, std::size_t> &edge) {
        int cluster_a = -1;
        int cluster_b = -1;
        if (!old_non_noise_cluster_id(edge.first, cluster_a) ||
            !old_non_noise_cluster_id(edge.second, cluster_b) ||
            cluster_a != cluster_b ||
            retained_connectivity_summary_clusters_.find(cluster_a) ==
                retained_connectivity_summary_clusters_.end()) {
          return false;
        }
        return single_old_component_removal_summary &&
               active_removed_ids.empty() &&
               summary_non_noise_edge_removals == 1 &&
               retained_bridge_edges_.find(ordered_pair(edge.first,
                                                        edge.second)) ==
                   retained_bridge_edges_.end();
      };

  auto old_articulation_summary_certifies = [&](std::size_t removed_id) {
    int cluster = -1;
    if (!old_non_noise_cluster_id(removed_id, cluster) ||
        retained_connectivity_summary_clusters_.find(cluster) ==
            retained_connectivity_summary_clusters_.end()) {
      return false;
    }
    return single_old_component_removal_summary &&
           active_removed_ids.size() == 1 &&
           active_removed_ids.find(removed_id) != active_removed_ids.end() &&
           summary_non_noise_vertex_removals == 1 &&
           summary_non_noise_edge_removals == 0 &&
           retained_articulation_vertices_.find(removed_id) ==
               retained_articulation_vertices_.end();
  };

  std::size_t component_witness_search_vertices = 0;
  const std::size_t max_local_witness_visits =
      std::max<std::size_t>(64, static_cast<std::size_t>(min_cluster_size) * 4);

  auto local_connection_preserved = [&](std::size_t a, std::size_t b) {
    std::size_t visited = 0;
    const bool connected =
        local_path_witness(retained_neighbors, a, b, max_local_witness_visits, &visited);
    component_witness_search_vertices += visited;
    return connected;
  };

  auto current_component_stays_noise = [&](std::size_t start) {
    if (!is_active(start))
      return true;

    std::unordered_set<std::size_t> visited;
    std::queue<std::size_t> q;
    visited.insert(start);
    q.push(start);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      int cluster = -1;
      bool is_noise = true;
      if (old_label(curr, cluster, is_noise)) {
        if (!is_noise && cluster >= 0)
          return false;
      } else if (active_added_ids.find(curr) == active_added_ids.end()) {
        return false;
      }
      if (visited.size() >= static_cast<std::size_t>(min_cluster_size))
        return false;

      for (std::size_t nb : retained_neighbors(curr)) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }
    return true;
  };

  auto added_noise_edge_preserved =
      [&](const std::pair<std::size_t, std::size_t> &edge) {
        return old_noise(edge.first) && old_noise(edge.second) &&
               current_component_stays_noise(edge.first);
      };

  bool partition_preserved =
      options_.experimental_component_witness && active_added_ids.empty();
  if (partition_preserved) {
    std::map<int, std::size_t> removed_by_cluster;
    for (std::size_t id : active_removed_ids) {
      int old_cluster = -1;
      bool old_noise = true;
      if (!old_label(id, old_cluster, old_noise))
        continue;
      if (old_noise || old_cluster < 0)
        continue;
      removed_by_cluster[old_cluster]++;
    }
    for (const auto &kv : removed_by_cluster) {
      const auto old_size_it = cluster_sizes_.find(kv.first);
      const std::size_t old_size =
          old_size_it == cluster_sizes_.end() ? 0 : old_size_it->second;
      const std::size_t remaining =
          old_size > kv.second ? old_size - kv.second : 0;
      if (remaining > 0 &&
          remaining < static_cast<std::size_t>(min_cluster_size)) {
        partition_preserved = false;
        break;
      }
    }
  }

  if (partition_preserved) {
    for (std::size_t removed_id : active_removed_ids) {
      int removed_cluster = -1;
      bool removed_noise = true;
      if (!old_label(removed_id, removed_cluster, removed_noise) ||
          removed_noise || removed_cluster < 0) {
        continue;
      }
      std::vector<std::size_t> active_neighbors;
      auto neighbors = removed_old_retained_neighbors.find(removed_id);
      if (neighbors != removed_old_retained_neighbors.end()) {
        for (std::size_t other : neighbors->second) {
          if (!is_active(other))
            continue;
          int other_cluster = -1;
          bool other_noise = true;
          if (!old_label(other, other_cluster, other_noise) || other_noise ||
              other_cluster != removed_cluster)
            continue;
          active_neighbors.push_back(other);
        }
      }
      std::sort(active_neighbors.begin(), active_neighbors.end());
      active_neighbors.erase(
          std::unique(active_neighbors.begin(), active_neighbors.end()),
          active_neighbors.end());
      if (active_neighbors.size() <= 1)
        continue;
      if (old_articulation_summary_certifies(removed_id))
        continue;
      for (std::size_t i = 1; i < active_neighbors.size(); ++i) {
        if (!local_connection_preserved(active_neighbors.front(),
                                        active_neighbors[i])) {
          partition_preserved = false;
          break;
        }
      }
      if (!partition_preserved)
        break;
    }
  }

  if (partition_preserved) {
    for (const auto &edge : removed_retained_edges) {
      if (active_removed_ids.find(edge.first) != active_removed_ids.end() ||
          active_removed_ids.find(edge.second) != active_removed_ids.end()) {
        continue;
      }
      if (old_noise(edge.first) && old_noise(edge.second))
        continue;
      if (!same_old_non_noise_cluster(edge.first, edge.second) ||
          (!old_bridge_summary_certifies(edge) &&
           !local_connection_preserved(edge.first, edge.second))) {
        partition_preserved = false;
        break;
      }
    }
  }

  if (partition_preserved) {
    for (const auto &edge : added_retained_edges) {
      if (added_noise_edge_preserved(edge))
        continue;
      if (!same_old_non_noise_cluster(edge.first, edge.second)) {
        partition_preserved = false;
        break;
      }
    }
  }

  if (partition_preserved) {
    if (changed_labels)
      *changed_labels = 0;
    if (relabel_scope_vertices)
      *relabel_scope_vertices = 0;
    if (relabel_search_vertices)
      *relabel_search_vertices = component_witness_search_vertices;
    std::set<int> summary_refresh_clusters = summary_affected_old_clusters;
    for (const auto &edge : added_retained_edges) {
      int cluster_a = -1;
      int cluster_b = -1;
      if (old_non_noise_cluster_id(edge.first, cluster_a) &&
          old_non_noise_cluster_id(edge.second, cluster_b) &&
          cluster_a == cluster_b) {
        summary_refresh_clusters.insert(cluster_a);
      }
    }
    refresh_cluster_stats_from_labels(false);
    if (connectivity_summary_enabled(options_))
      refresh_retained_connectivity_summary_for_clusters(
          summary_refresh_clusters);
    if (!skip_bucket_refresh)
      refresh_bucket_index_and_stats();
    return;
  }

  bool handled_cluster_merges =
      active_added_ids.empty() && active_removed_ids.empty() &&
      removed_retained_edges.empty() && !added_retained_edges.empty();
  std::map<int, std::set<int>> merge_adjacency;
  if (handled_cluster_merges) {
    for (const auto &edge : added_retained_edges) {
      int cluster_a = -1;
      int cluster_b = -1;
      bool noise_a = true;
      bool noise_b = true;
      if (!old_label(edge.first, cluster_a, noise_a) ||
          !old_label(edge.second, cluster_b, noise_b) || noise_a || noise_b ||
          cluster_a < 0 || cluster_b < 0) {
        handled_cluster_merges = false;
        break;
      }
      if (cluster_a == cluster_b)
        continue;
      merge_adjacency[cluster_a].insert(cluster_b);
      merge_adjacency[cluster_b].insert(cluster_a);
    }
    if (merge_adjacency.empty())
      handled_cluster_merges = false;
  }
  if (handled_cluster_merges) {
    std::set<int> visited_clusters;
    std::set<int> summary_refresh_clusters;
    std::size_t updated_vertices = 0;

    for (const auto &kv : merge_adjacency) {
      const int start_cluster = kv.first;
      if (!visited_clusters.insert(start_cluster).second)
        continue;

      std::vector<int> group;
      std::queue<int> q;
      q.push(start_cluster);
      while (!q.empty()) {
        const int current = q.front();
        q.pop();
        group.push_back(current);
        auto adj_it = merge_adjacency.find(current);
        if (adj_it == merge_adjacency.end())
          continue;
        for (int nb : adj_it->second) {
          if (visited_clusters.insert(nb).second)
            q.push(nb);
        }
      }

      int survivor = group.front();
      std::size_t survivor_size = 0;
      bool have_all_members = true;
      for (int cluster_id : group) {
        summary_refresh_clusters.insert(cluster_id);
        auto size_it = cluster_sizes_.find(cluster_id);
        auto members_it = cluster_members_.find(cluster_id);
        if (size_it == cluster_sizes_.end() ||
            members_it == cluster_members_.end()) {
          have_all_members = false;
          break;
        }
        if (size_it->second > survivor_size ||
            (size_it->second == survivor_size && cluster_id < survivor)) {
          survivor = cluster_id;
          survivor_size = size_it->second;
        }
      }
      if (!have_all_members) {
        handled_cluster_merges = false;
        break;
      }
      summary_refresh_clusters.insert(survivor);

      auto &survivor_members = cluster_members_[survivor];
      for (int cluster_id : group) {
        if (cluster_id == survivor)
          continue;
        auto members_it = cluster_members_.find(cluster_id);
        if (members_it == cluster_members_.end()) {
          handled_cluster_merges = false;
          break;
        }
        std::vector<std::size_t> members(members_it->second.begin(),
                                         members_it->second.end());
        for (std::size_t id : members) {
          auto idx = id_to_index_.find(id);
          if (idx == id_to_index_.end())
            continue;
          PointRecord &record = records_[idx->second];
          if (!record.active)
            continue;
          if (record.cluster_id != survivor || record.is_noise) {
            record.cluster_id = survivor;
            record.is_noise = false;
            updated_vertices++;
          }
          survivor_members.insert(id);
        }
        cluster_members_.erase(cluster_id);
        cluster_sizes_.erase(cluster_id);
      }
      if (!handled_cluster_merges)
        break;
      cluster_sizes_[survivor] = survivor_members.size();
    }

    if (handled_cluster_merges) {
      stats_.num_clusters = cluster_sizes_.size();
      if (connectivity_summary_enabled(options_))
        refresh_retained_connectivity_summary_for_clusters(
            summary_refresh_clusters);
      if (changed_labels)
        *changed_labels = updated_vertices;
      if (relabel_scope_vertices)
        *relabel_scope_vertices = updated_vertices;
      if (relabel_search_vertices)
        *relabel_search_vertices = 0;
      if (!skip_bucket_refresh)
        refresh_bucket_index_and_stats();
      return;
    }
  }

  bool handled_added_vertices = !active_added_ids.empty() &&
                                active_removed_ids.empty();
  if (handled_added_vertices) {
    for (const auto &edge : removed_retained_edges) {
      if (old_noise(edge.first) && old_noise(edge.second))
        continue;
      if (!reachable_in_adjacency(retained_neighbors, edge.first, edge.second)) {
        handled_added_vertices = false;
        break;
      }
    }
  }
  if (handled_added_vertices) {
    for (const auto &edge : added_retained_edges) {
      const bool first_new =
          active_added_ids.find(edge.first) != active_added_ids.end();
      const bool second_new =
          active_added_ids.find(edge.second) != active_added_ids.end();
      if (first_new || second_new)
        continue;
      if (added_noise_edge_preserved(edge))
        continue;
      if (!same_old_non_noise_cluster(edge.first, edge.second) &&
          !reachable_in_adjacency_without_edge(retained_neighbors, edge.first, edge.second)) {
        handled_added_vertices = false;
        break;
      }
    }
  }
  if (handled_added_vertices) {
    std::set<int> used_cluster_ids;
    for (const auto &kv : cluster_sizes_)
      used_cluster_ids.insert(kv.first);
    int next_cluster_id =
        used_cluster_ids.empty() ? 0 : *used_cluster_ids.rbegin() + 1;
    auto allocate_cluster_id = [&used_cluster_ids, &next_cluster_id]() {
      while (used_cluster_ids.find(next_cluster_id) != used_cluster_ids.end())
        next_cluster_id++;
      const int id = next_cluster_id++;
      used_cluster_ids.insert(id);
      return id;
    };

    const int min_cluster_size =
        options_.min_cluster_size > 0 ? options_.min_cluster_size
                                      : default_min_cluster_size();
    std::unordered_set<std::size_t> visited_new;
    std::size_t relabeled_new_vertices = 0;
    for (std::size_t start : active_added_ids) {
      if (visited_new.find(start) != visited_new.end())
        continue;

      std::vector<std::size_t> new_component;
      std::set<int> adjacent_clusters;
      bool touches_old_noise = false;
      std::queue<std::size_t> q;
      q.push(start);
      visited_new.insert(start);
      while (!q.empty()) {
        const std::size_t curr = q.front();
        q.pop();
        new_component.push_back(curr);
        for (std::size_t nb : retained_neighbors(curr)) {
          if (active_added_ids.find(nb) != active_added_ids.end()) {
            if (visited_new.insert(nb).second)
              q.push(nb);
            continue;
          }
          int old_cluster = -1;
          bool old_noise = true;
          if (!old_label(nb, old_cluster, old_noise)) {
            handled_added_vertices = false;
            break;
          }
          if (old_noise || old_cluster < 0)
            touches_old_noise = true;
          else
            adjacent_clusters.insert(old_cluster);
        }
        if (!handled_added_vertices)
          break;
      }
      if (!handled_added_vertices)
        break;

      bool is_noise = false;
      int cluster_id = -1;
      if (adjacent_clusters.empty()) {
        is_noise =
            static_cast<int>(new_component.size()) < min_cluster_size;
        if (!is_noise)
          cluster_id = allocate_cluster_id();
      } else if (adjacent_clusters.size() == 1 && !touches_old_noise) {
        cluster_id = *adjacent_clusters.begin();
        is_noise = false;
      } else {
        handled_added_vertices = false;
        break;
      }

      for (std::size_t id : new_component) {
        PointRecord &p = records_[id_to_index_.at(id)];
        p.cluster_id = cluster_id;
        p.is_noise = is_noise;
        relabeled_new_vertices++;
      }
    }

    if (handled_added_vertices) {
      if (changed_labels) {
        *changed_labels = 0;
        for (std::size_t id : active_added_ids) {
          auto idx = id_to_index_.find(id);
          if (idx == id_to_index_.end())
            continue;
          const PointRecord &record = records_[idx->second];
          if (!record.active)
            continue;
          if (record.cluster_id != -1 || !record.is_noise)
            (*changed_labels)++;
        }
      }
      if (relabel_scope_vertices)
        *relabel_scope_vertices = relabeled_new_vertices;
      if (relabel_search_vertices)
        *relabel_search_vertices = relabeled_new_vertices;
      std::set<int> summary_refresh_clusters = summary_affected_old_clusters;
      for (const auto &edge : added_retained_edges) {
        int cluster_a = -1;
        int cluster_b = -1;
        if (old_non_noise_cluster_id(edge.first, cluster_a) &&
            old_non_noise_cluster_id(edge.second, cluster_b) &&
            cluster_a == cluster_b) {
          summary_refresh_clusters.insert(cluster_a);
        }
      }
      for (std::size_t id : active_added_ids) {
        auto idx = id_to_index_.find(id);
        if (idx == id_to_index_.end())
          continue;
        const PointRecord &record = records_[idx->second];
        if (record.active && !record.is_noise && record.cluster_id >= 0)
          summary_refresh_clusters.insert(record.cluster_id);
      }
      refresh_cluster_stats_from_labels(false);
      if (connectivity_summary_enabled(options_))
        refresh_retained_connectivity_summary_for_clusters(
            summary_refresh_clusters);
      if (!skip_bucket_refresh)
        refresh_bucket_index_and_stats();
      return;
    }
  }

  bool handled_small_side_split =
      options_.experimental_small_side_split && active_removed_ids.empty() &&
      !removed_retained_edges.empty() &&
      summary_non_noise_edge_removals == removed_retained_edges.size() &&
      summary_non_noise_vertex_removals == 0 &&
      summary_affected_old_clusters.size() == 1;
  const int split_old_cluster =
      summary_affected_old_clusters.empty() ? -1
                                            : *summary_affected_old_clusters.begin();

  if (handled_small_side_split) {
    for (const auto &edge : added_retained_edges) {
      const bool first_new =
          active_added_ids.find(edge.first) != active_added_ids.end();
      const bool second_new =
          active_added_ids.find(edge.second) != active_added_ids.end();
      if (first_new && second_new)
        continue;

      int cluster_a = -1;
      int cluster_b = -1;
      if (first_new || second_new) {
        const std::size_t old_id = first_new ? edge.second : edge.first;
        if (!old_non_noise_cluster_id(old_id, cluster_a) ||
            cluster_a != split_old_cluster) {
          handled_small_side_split = false;
          break;
        }
        continue;
      }

      if (!old_non_noise_cluster_id(edge.first, cluster_a) ||
          !old_non_noise_cluster_id(edge.second, cluster_b) ||
          cluster_a != split_old_cluster || cluster_b != split_old_cluster) {
        handled_small_side_split = false;
        break;
      }
    }
  }

  auto bounded_component_from = [&](std::size_t start,
                                    std::vector<std::size_t> &component,
                                    std::size_t &visited_count) {
    component.clear();
    visited_count = 0;
    if (!is_active(start))
      return false;

    std::unordered_set<std::size_t> visited;
    std::queue<std::size_t> q;
    visited.insert(start);
    q.push(start);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component.push_back(curr);
      visited_count = std::max(visited_count, visited.size());
      if (visited.size() > kMaxSmallSideSplitSearch)
        return false;

      for (std::size_t nb : retained_neighbors(curr)) {
        if (visited.insert(nb).second) {
          if (visited.size() > kMaxSmallSideSplitSearch)
            return false;
          q.push(nb);
        }
      }
    }
    return true;
  };

  bool handled_point_delete_split =
      options_.experimental_small_side_split && active_added_ids.empty() &&
      active_removed_ids.size() == 1 && summary_non_noise_vertex_removals == 1 &&
      summary_affected_old_clusters.size() == 1;

  if (handled_point_delete_split) {
    const std::size_t removed_id = *active_removed_ids.begin();
    int removed_cluster = -1;
    if (!old_non_noise_cluster_id(removed_id, removed_cluster) ||
        removed_cluster != split_old_cluster) {
      handled_point_delete_split = false;
    }
  }

  if (handled_point_delete_split) {
    const std::size_t removed_id = *active_removed_ids.begin();
    for (const auto &edge : removed_retained_edges) {
      const bool first_removed = edge.first == removed_id;
      const bool second_removed = edge.second == removed_id;
      if (first_removed || second_removed) {
        const std::size_t other = first_removed ? edge.second : edge.first;
        int other_cluster = -1;
        if (!old_non_noise_cluster_id(other, other_cluster) ||
            other_cluster != split_old_cluster) {
          handled_point_delete_split = false;
          break;
        }
        continue;
      }

      int cluster_a = -1;
      int cluster_b = -1;
      if (old_noise(edge.first) && old_noise(edge.second) &&
          current_component_stays_noise(edge.first)) {
        continue;
      }
      if (!old_non_noise_cluster_id(edge.first, cluster_a) ||
          !old_non_noise_cluster_id(edge.second, cluster_b) ||
          cluster_a != split_old_cluster || cluster_b != split_old_cluster) {
        handled_point_delete_split = false;
        break;
      }
    }
  }

  if (handled_point_delete_split) {
    for (const auto &edge : added_retained_edges) {
      int cluster_a = -1;
      int cluster_b = -1;
      if (added_noise_edge_preserved(edge))
        continue;
      if (!old_non_noise_cluster_id(edge.first, cluster_a) ||
          !old_non_noise_cluster_id(edge.second, cluster_b) ||
          cluster_a != split_old_cluster || cluster_b != split_old_cluster) {
        handled_point_delete_split = false;
        break;
      }
    }
  }

  if (handled_point_delete_split) {
    const std::size_t removed_id = *active_removed_ids.begin();
    const auto old_size_it = cluster_sizes_.find(split_old_cluster);
    const std::size_t old_cluster_size =
        old_size_it == cluster_sizes_.end() ? 0 : old_size_it->second;
    const std::size_t remaining_old_size =
        old_cluster_size > 0 ? old_cluster_size - 1 : 0;
    if (old_cluster_size == 0 || remaining_old_size == 0)
      handled_point_delete_split = false;

    std::vector<std::vector<std::size_t>> exact_components;
    std::unordered_set<std::size_t> exact_ids;
    std::unordered_set<std::size_t> unknown_partial_ids;
    bool have_unknown_component = false;
    std::size_t small_side_search = 0;

    if (handled_point_delete_split) {
      std::vector<std::size_t> endpoints;
      auto append_endpoint = [&](std::size_t id) {
        if (id == removed_id || !is_active(id)) {
          return;
        }
        int cluster = -1;
        if (old_non_noise_cluster_id(id, cluster) &&
            cluster == split_old_cluster) {
          endpoints.push_back(id);
        }
      };

      auto neighbors = removed_old_retained_neighbors.find(removed_id);
      if (neighbors != removed_old_retained_neighbors.end()) {
        for (std::size_t nb : neighbors->second)
          append_endpoint(nb);
      }
      for (const auto &edge : removed_retained_edges) {
        append_endpoint(edge.first);
        append_endpoint(edge.second);
      }
      for (const auto &edge : added_retained_edges) {
        append_endpoint(edge.first);
        append_endpoint(edge.second);
      }
      std::sort(endpoints.begin(), endpoints.end());
      endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                      endpoints.end());
      if (endpoints.empty()) {
        handled_point_delete_split = false;
      }

      for (std::size_t endpoint : endpoints) {
        if (!handled_point_delete_split)
          break;
        if (exact_ids.find(endpoint) != exact_ids.end() ||
            unknown_partial_ids.find(endpoint) != unknown_partial_ids.end()) {
          continue;
        }

        std::vector<std::size_t> component;
        std::size_t visited_count = 0;
        const bool complete =
            bounded_component_from(endpoint, component, visited_count);
        small_side_search += visited_count;
        if (!complete) {
          if (have_unknown_component) {
            handled_point_delete_split = false;
            break;
          }
          have_unknown_component = true;
          unknown_partial_ids.insert(component.begin(), component.end());
          continue;
        }

        for (std::size_t id : component)
          exact_ids.insert(id);
        exact_components.push_back(std::move(component));
      }
    }

    if (handled_point_delete_split) {
      std::size_t exact_old_ids = 0;
      for (const auto &component : exact_components) {
        for (std::size_t id : component) {
          int cluster = -1;
          if (!old_non_noise_cluster_id(id, cluster) ||
              cluster != split_old_cluster) {
            handled_point_delete_split = false;
            break;
          }
          exact_old_ids++;
        }
        if (!handled_point_delete_split)
          break;
      }

      if (handled_point_delete_split) {
        if (!have_unknown_component && exact_old_ids != remaining_old_size) {
          handled_point_delete_split = false;
        } else if (have_unknown_component && exact_old_ids >= remaining_old_size) {
          handled_point_delete_split = false;
        }
      }
      if (handled_point_delete_split) {
        const std::size_t unknown_old_size =
            remaining_old_size > exact_old_ids ? remaining_old_size - exact_old_ids
                                               : 0;
        if (have_unknown_component &&
            unknown_old_size < static_cast<std::size_t>(min_cluster_size)) {
          handled_point_delete_split = false;
        }
      }
    }

    if (handled_point_delete_split) {
      if (exact_components.size() == 1 && !have_unknown_component &&
          exact_components.front().size() == remaining_old_size) {
        if (changed_labels)
          *changed_labels = 0;
        if (relabel_scope_vertices)
          *relabel_scope_vertices = 0;
        if (relabel_search_vertices)
          *relabel_search_vertices = small_side_search;
        last_small_side_split_used_ = true;
        last_point_delete_split_used_ = true;
        refresh_cluster_stats_from_labels(false);
        if (connectivity_summary_enabled(options_))
          refresh_retained_connectivity_summary_for_clusters(
              {split_old_cluster});
        if (!skip_bucket_refresh)
          refresh_bucket_index_and_stats();
        return;
      }

      std::set<int> used_cluster_ids;
      for (const auto &kv : cluster_sizes_)
        used_cluster_ids.insert(kv.first);
      int next_cluster_id =
          used_cluster_ids.empty() ? 0 : *used_cluster_ids.rbegin() + 1;
      while (used_cluster_ids.find(next_cluster_id) != used_cluster_ids.end())
        next_cluster_id++;

      std::size_t keep_old_index = exact_components.size();
      if (!have_unknown_component) {
        std::size_t best_size = 0;
        for (std::size_t i = 0; i < exact_components.size(); ++i) {
          if (exact_components[i].size() > best_size) {
            best_size = exact_components[i].size();
            keep_old_index = i;
          }
        }
        if (best_size < static_cast<std::size_t>(min_cluster_size))
          keep_old_index = exact_components.size();
      }

      std::size_t changed = 0;
      std::size_t label_update_scope = 0;
      std::set<int> summary_refresh_clusters{split_old_cluster};
      for (std::size_t i = 0; i < exact_components.size(); ++i) {
        const bool keep_old = !have_unknown_component && i == keep_old_index;
        const bool component_is_noise =
            exact_components[i].size() < static_cast<std::size_t>(min_cluster_size);
        int component_cluster = -1;
        if (keep_old) {
          component_cluster = split_old_cluster;
        } else if (!component_is_noise) {
          component_cluster = next_cluster_id++;
          while (used_cluster_ids.find(component_cluster) != used_cluster_ids.end())
            component_cluster = next_cluster_id++;
          used_cluster_ids.insert(component_cluster);
          summary_refresh_clusters.insert(component_cluster);
        }

        const bool target_noise = !keep_old && component_is_noise;
        for (std::size_t id : exact_components[i]) {
          PointRecord &record = records_[id_to_index_.at(id)];
          const bool needs_write = record.cluster_id != component_cluster ||
                                   record.is_noise != target_noise;
          if (needs_write) {
            record.cluster_id = component_cluster;
            record.is_noise = target_noise;
            label_update_scope++;
            changed++;
          }
        }
      }

      if (changed_labels)
        *changed_labels = changed;
      if (relabel_scope_vertices)
        *relabel_scope_vertices = label_update_scope;
      if (relabel_search_vertices)
        *relabel_search_vertices = small_side_search;
      last_small_side_split_used_ = true;
      last_point_delete_split_used_ = true;
      refresh_cluster_stats_from_labels(false);
      if (connectivity_summary_enabled(options_))
        refresh_retained_connectivity_summary_for_clusters(
            summary_refresh_clusters);
      if (!skip_bucket_refresh)
        refresh_bucket_index_and_stats();
      return;
    }
  }

  if (handled_small_side_split) {
    const auto old_size_it = cluster_sizes_.find(split_old_cluster);
    const std::size_t old_cluster_size =
        old_size_it == cluster_sizes_.end() ? 0 : old_size_it->second;
    if (old_cluster_size == 0)
      handled_small_side_split = false;

    std::vector<std::vector<std::size_t>> exact_components;
    std::unordered_set<std::size_t> exact_ids;
    std::unordered_set<std::size_t> unknown_partial_ids;
    bool have_unknown_component = false;
    std::size_t small_side_search = 0;

    if (handled_small_side_split) {
      std::vector<std::size_t> endpoints;
      endpoints.reserve(removed_retained_edges.size() * 2);
      for (const auto &edge : removed_retained_edges) {
        endpoints.push_back(edge.first);
        endpoints.push_back(edge.second);
      }
      endpoints.insert(endpoints.end(), active_added_ids.begin(),
                       active_added_ids.end());
      std::sort(endpoints.begin(), endpoints.end());
      endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                      endpoints.end());

      for (std::size_t endpoint : endpoints) {
        if (exact_ids.find(endpoint) != exact_ids.end() ||
            unknown_partial_ids.find(endpoint) != unknown_partial_ids.end()) {
          continue;
        }

        std::vector<std::size_t> component;
        std::size_t visited_count = 0;
        const bool complete =
            bounded_component_from(endpoint, component, visited_count);
        small_side_search += visited_count;
        if (!complete) {
          if (have_unknown_component) {
            handled_small_side_split = false;
            break;
          }
          have_unknown_component = true;
          unknown_partial_ids.insert(component.begin(), component.end());
          continue;
        }

        for (std::size_t id : component)
          exact_ids.insert(id);
        exact_components.push_back(std::move(component));
      }
    }

    if (handled_small_side_split) {
      std::size_t exact_old_ids = 0;
      std::size_t exact_added_ids = 0;
      for (const auto &component : exact_components) {
        for (std::size_t id : component) {
          if (active_added_ids.find(id) != active_added_ids.end()) {
            exact_added_ids++;
            continue;
          }
          int cluster = -1;
          if (!old_non_noise_cluster_id(id, cluster) ||
              cluster != split_old_cluster) {
            handled_small_side_split = false;
            break;
          }
          exact_old_ids++;
        }
        if (!handled_small_side_split)
          break;
      }

      for (std::size_t id : active_added_ids) {
        if (exact_ids.find(id) == exact_ids.end()) {
          handled_small_side_split = false;
          break;
        }
      }

      if (handled_small_side_split) {
        if (!have_unknown_component &&
            (exact_old_ids != old_cluster_size ||
             exact_added_ids != active_added_ids.size())) {
          handled_small_side_split = false;
        } else if (have_unknown_component &&
                   exact_old_ids >= old_cluster_size) {
          handled_small_side_split = false;
        }
      }
      if (handled_small_side_split) {
        const std::size_t remaining_old_size =
            old_cluster_size > exact_old_ids ? old_cluster_size - exact_old_ids
                                             : 0;
        if (have_unknown_component &&
            remaining_old_size < static_cast<std::size_t>(min_cluster_size)) {
          handled_small_side_split = false;
        }
      }
    }

    if (handled_small_side_split) {
      if (exact_components.size() == 1 && !have_unknown_component &&
          exact_components.front().size() == old_cluster_size) {
        if (changed_labels)
          *changed_labels = 0;
        if (relabel_scope_vertices)
          *relabel_scope_vertices = 0;
        if (relabel_search_vertices)
          *relabel_search_vertices = small_side_search;
        last_small_side_split_used_ = true;
        std::set<int> summary_refresh_clusters{split_old_cluster};
        refresh_cluster_stats_from_labels(false);
        if (connectivity_summary_enabled(options_))
          refresh_retained_connectivity_summary_for_clusters(
              summary_refresh_clusters);
        if (!skip_bucket_refresh)
          refresh_bucket_index_and_stats();
        return;
      }

      std::set<int> used_cluster_ids;
      for (const auto &kv : cluster_sizes_)
        used_cluster_ids.insert(kv.first);
      int next_cluster_id =
          used_cluster_ids.empty() ? 0 : *used_cluster_ids.rbegin() + 1;
      while (used_cluster_ids.find(next_cluster_id) != used_cluster_ids.end())
        next_cluster_id++;

      std::size_t keep_old_index = exact_components.size();
      if (!have_unknown_component) {
        std::size_t best_size = 0;
        for (std::size_t i = 0; i < exact_components.size(); ++i) {
          if (exact_components[i].size() > best_size) {
            best_size = exact_components[i].size();
            keep_old_index = i;
          }
        }
        if (best_size < static_cast<std::size_t>(min_cluster_size))
          keep_old_index = exact_components.size();
      }

      std::size_t changed = 0;
      std::size_t label_update_scope = 0;
      std::set<int> summary_refresh_clusters{split_old_cluster};
      for (std::size_t i = 0; i < exact_components.size(); ++i) {
        const bool keep_old = !have_unknown_component && i == keep_old_index;
        const bool component_is_noise =
            exact_components[i].size() < static_cast<std::size_t>(min_cluster_size);
        int component_cluster = -1;
        if (keep_old) {
          component_cluster = split_old_cluster;
        } else if (!component_is_noise) {
          component_cluster = next_cluster_id++;
          while (used_cluster_ids.find(component_cluster) != used_cluster_ids.end())
            component_cluster = next_cluster_id++;
          used_cluster_ids.insert(component_cluster);
          summary_refresh_clusters.insert(component_cluster);
        }

        const bool target_noise = !keep_old && component_is_noise;
        for (std::size_t id : exact_components[i]) {
          PointRecord &record = records_[id_to_index_.at(id)];
          const bool needs_write = record.cluster_id != component_cluster ||
                                   record.is_noise != target_noise;
          if (needs_write) {
            record.cluster_id = component_cluster;
            record.is_noise = target_noise;
            label_update_scope++;
            changed++;
          }
        }
      }

      for (const auto &edge : added_retained_edges) {
        int cluster_a = -1;
        int cluster_b = -1;
        if (old_non_noise_cluster_id(edge.first, cluster_a) &&
            old_non_noise_cluster_id(edge.second, cluster_b) &&
            cluster_a == cluster_b) {
          summary_refresh_clusters.insert(cluster_a);
        }
      }

      if (changed_labels)
        *changed_labels = changed;
      if (relabel_scope_vertices)
        *relabel_scope_vertices = label_update_scope;
      if (relabel_search_vertices)
        *relabel_search_vertices = small_side_search;
      last_small_side_split_used_ = true;
      refresh_cluster_stats_from_labels(false);
      if (connectivity_summary_enabled(options_))
        refresh_retained_connectivity_summary_for_clusters(
            summary_refresh_clusters);
      if (!skip_bucket_refresh)
        refresh_bucket_index_and_stats();
      return;
    }
  }

  std::set<std::size_t> seeds = seed_ids;
  for (const auto &edge : removed_retained_edges) {
    seeds.insert(edge.first);
    seeds.insert(edge.second);
  }
  for (const auto &edge : added_retained_edges) {
    seeds.insert(edge.first);
    seeds.insert(edge.second);
  }

  std::unordered_set<std::size_t> local_ids;
  std::vector<std::vector<std::size_t>> local_components;
  for (std::size_t seed : seeds) {
    if (local_ids.find(seed) != local_ids.end())
      continue;
    if (!is_active(seed))
      continue;

    std::vector<std::size_t> component;
    std::queue<std::size_t> q;
    q.push(seed);
    local_ids.insert(seed);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component.push_back(curr);
      for (std::size_t nb : retained_neighbors(curr)) {
        if (local_ids.insert(nb).second)
          q.push(nb);
      }
    }
    local_components.push_back(component);
  }

  if (relabel_search_vertices)
    *relabel_search_vertices = local_ids.size();

  std::set<int> used_cluster_ids;
  for (const auto &record : records_) {
    if (!record.active || record.is_noise || record.cluster_id < 0)
      continue;
    if (local_ids.find(record.id) == local_ids.end())
      used_cluster_ids.insert(record.cluster_id);
  }

  int next_cluster_id = 0;
  if (!used_cluster_ids.empty())
    next_cluster_id = *used_cluster_ids.rbegin() + 1;
  auto allocate_cluster_id = [&used_cluster_ids, &next_cluster_id]() {
    while (used_cluster_ids.find(next_cluster_id) != used_cluster_ids.end())
      next_cluster_id++;
    const int id = next_cluster_id++;
    used_cluster_ids.insert(id);
    return id;
  };

  struct LocalComponentPlan {
    std::vector<std::size_t> ids;
    std::map<int, std::size_t> reusable_old_clusters;
    bool is_noise = true;
    int target_cluster = -1;
  };

  std::vector<LocalComponentPlan> plans;
  plans.reserve(local_components.size());
  std::map<int, std::pair<std::size_t, std::size_t>> best_plan_for_cluster;
  for (const auto &component : local_components) {
    LocalComponentPlan plan;
    plan.ids = component;
    plan.is_noise = static_cast<int>(component.size()) < min_cluster_size;
    const std::size_t plan_index = plans.size();

    if (!plan.is_noise) {
      for (std::size_t id : component) {
        int old_cluster = -1;
        bool old_noise = true;
        if (!old_label(id, old_cluster, old_noise))
          continue;
        if (!old_noise && old_cluster >= 0 &&
            used_cluster_ids.find(old_cluster) == used_cluster_ids.end()) {
          plan.reusable_old_clusters[old_cluster]++;
        }
      }

      for (const auto &candidate : plan.reusable_old_clusters) {
        auto best = best_plan_for_cluster.find(candidate.first);
        if (best == best_plan_for_cluster.end() ||
            candidate.second > best->second.second ||
            (candidate.second == best->second.second &&
             plan.ids.size() > plans[best->second.first].ids.size())) {
          best_plan_for_cluster[candidate.first] =
              {plan_index, candidate.second};
        }
      }
    }

    plans.push_back(std::move(plan));
  }

  std::vector<std::vector<int>> keep_candidates(plans.size());
  for (const auto &kv : best_plan_for_cluster) {
    keep_candidates[kv.second.first].push_back(kv.first);
    used_cluster_ids.insert(kv.first);
  }

  for (std::size_t i = 0; i < plans.size(); ++i) {
    LocalComponentPlan &plan = plans[i];
    if (plan.is_noise) {
      plan.target_cluster = -1;
      continue;
    }

    if (!keep_candidates[i].empty()) {
      plan.target_cluster = *std::max_element(
          keep_candidates[i].begin(), keep_candidates[i].end(),
          [&plan](int a, int b) {
            const std::size_t count_a = plan.reusable_old_clusters[a];
            const std::size_t count_b = plan.reusable_old_clusters[b];
            if (count_a != count_b)
              return count_a < count_b;
            return a > b;
          });
    } else {
      plan.target_cluster = allocate_cluster_id();
    }
  }

  std::set<int> summary_refresh_clusters = summary_affected_old_clusters;
  for (std::size_t id : local_ids) {
    int old_cluster = -1;
    if (old_non_noise_cluster_id(id, old_cluster))
      summary_refresh_clusters.insert(old_cluster);
  }
  for (const auto &plan : plans) {
    if (!plan.is_noise && plan.target_cluster >= 0)
      summary_refresh_clusters.insert(plan.target_cluster);
  }

  std::size_t label_update_scope = 0;
  std::size_t changed = 0;
  for (const auto &plan : plans) {
    for (std::size_t id : plan.ids) {
      auto idx = id_to_index_.find(id);
      if (idx == id_to_index_.end())
        continue;
      PointRecord &record = records_[idx->second];
      if (!record.active)
        continue;

      int old_cluster = -1;
      bool old_noise = true;
      const bool had_old_label = old_label(id, old_cluster, old_noise);
      const bool logical_change =
          had_old_label ? (old_cluster != plan.target_cluster ||
                           old_noise != plan.is_noise)
                        : (plan.target_cluster != -1 || !plan.is_noise);
      const bool needs_write = record.cluster_id != plan.target_cluster ||
                               record.is_noise != plan.is_noise;
      if (needs_write) {
        record.cluster_id = plan.target_cluster;
        record.is_noise = plan.is_noise;
        label_update_scope++;
      }
      if (logical_change)
        changed++;
    }
  }

  if (changed_labels)
    *changed_labels = changed;
  if (relabel_scope_vertices)
    *relabel_scope_vertices = label_update_scope;

  refresh_cluster_stats_from_labels(false);
  if (connectivity_summary_enabled(options_))
    refresh_retained_connectivity_summary_for_clusters(
        summary_refresh_clusters);

  if (!skip_bucket_refresh)
    refresh_bucket_index_and_stats();
}

void DelaunayClusterer::rebuild_triangulation() {
  dt_.clear();
  vertex_handles_.clear();
  std::vector<std::pair<Point, std::size_t>> inserts;
  for (const auto &record : records_) {
    if (record.active) {
      inserts.push_back({Point(record.x, record.y), record.id});
    }
  }
  dt_.insert(inserts.begin(), inserts.end());
  for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
       ++v) {
    vertex_handles_[v->info()] = v;
  }
}

void DelaunayClusterer::extract_edges_and_scales() {
  last_local_edge_refresh_tracked_ = false;
  last_bucket_removed_edge_ids_.clear();
  last_bucket_upsert_edge_ids_.clear();
  last_removed_retained_edges_.clear();
  last_added_retained_edges_.clear();
  retained_bridge_edges_.clear();
  retained_articulation_vertices_.clear();
  retained_bridge_edges_by_cluster_.clear();
  retained_articulation_vertices_by_cluster_.clear();
  retained_connectivity_summary_clusters_.clear();
  edges_.clear();
  local_scale_.clear();
  stats_ = ClusterStats{};
  stats_.point_count = active_ids().size();
  if (stats_.point_count < 2) {
    next_edge_id_ = 0;
    rebuild_edge_lookup();
    return;
  }

  std::unordered_map<EdgeKey, bool, EdgeKeyHash> seen;
  std::unordered_map<std::size_t, std::vector<double>> incident_lengths;
  std::unordered_map<std::size_t, const PointRecord *> by_id;
  for (const auto &record : records_) {
    if (record.active)
      by_id[record.id] = &record;
  }

  for (auto e = dt_.finite_edges_begin(); e != dt_.finite_edges_end(); ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);
    std::size_t a = v1->info();
    std::size_t b = v2->info();
    if (a == b)
      continue;
    if (a > b)
      std::swap(a, b);
    EdgeKey key{a, b};
    if (seen.find(key) != seen.end())
      continue;
    seen[key] = true;

    const auto *ra = by_id[a];
    const auto *rb = by_id[b];
    const double len = distance_xy(ra->x, ra->y, rb->x, rb->y);
    EdgeRecord edge;
    edge.id = edges_.size();
    edge.u = a;
    edge.v = b;
    edge.length = len;
    edges_.push_back(edge);
    incident_lengths[a].push_back(len);
    incident_lengths[b].push_back(len);
  }
  std::sort(edges_.begin(), edges_.end(),
            [](const EdgeRecord &lhs, const EdgeRecord &rhs) {
              if (lhs.u != rhs.u)
                return lhs.u < rhs.u;
              if (lhs.v != rhs.v)
                return lhs.v < rhs.v;
              return lhs.length < rhs.length;
            });
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i].id = i;
  next_edge_id_ = edges_.size();

  std::vector<double> all_lengths;
  all_lengths.reserve(edges_.size());
  for (const auto &edge : edges_)
    all_lengths.push_back(edge.length);
  std::sort(all_lengths.begin(), all_lengths.end());
  const double fallback_scale =
      std::max(median_sorted(all_lengths), kMinPadding);

  for (const auto &record : records_) {
    if (!record.active)
      continue;
    auto lengths = incident_lengths[record.id];
    if (lengths.empty()) {
      local_scale_[record.id] = fallback_scale;
    } else {
      std::sort(lengths.begin(), lengths.end());
      local_scale_[record.id] =
          std::max(median_sorted(lengths), kMinPadding);
    }
  }
  if (normalized_threshold_mode(options_) == "knn_median") {
    std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
    for (const auto &record : records_) {
      if (record.active)
        adj[record.id] = {};
    }
    for (const auto &edge : edges_) {
      adj[edge.u].push_back(edge.v);
      adj[edge.v].push_back(edge.u);
    }
    for (auto &entry : adj)
      std::sort(entry.second.begin(), entry.second.end());
    const int k = options_.knn_scale_k > 0 ? options_.knn_scale_k
                                           : default_min_cluster_size();
    const std::size_t max_neighbors =
        std::max<std::size_t>(1, static_cast<std::size_t>(k));
    for (const auto &record : records_) {
      if (!record.active)
        continue;
      std::vector<double> distances;
      distances.reserve(max_neighbors);
      std::unordered_set<std::size_t> visited;
      std::queue<std::size_t> q;
      visited.insert(record.id);
      q.push(record.id);
      while (!q.empty() && distances.size() < max_neighbors) {
        const std::size_t curr = q.front();
        q.pop();
        auto adj_it = adj.find(curr);
        if (adj_it == adj.end())
          continue;
        for (std::size_t nb : adj_it->second) {
          if (!visited.insert(nb).second)
            continue;
          const auto nb_idx = id_to_index_.find(nb);
          if (nb_idx != id_to_index_.end() && records_[nb_idx->second].active) {
            distances.push_back(distance_xy(record.x, record.y,
                                            records_[nb_idx->second].x,
                                            records_[nb_idx->second].y));
            if (distances.size() >= max_neighbors)
              break;
          }
          q.push(nb);
        }
      }
      if (!distances.empty()) {
        std::sort(distances.begin(), distances.end());
        local_scale_[record.id] =
            std::max(median_sorted(distances), kMinPadding);
      } else {
        local_scale_[record.id] = fallback_scale;
      }
    }
  }
  rebuild_edge_lookup();
}

void DelaunayClusterer::choose_threshold_and_prune() {
  std::vector<double> scores;
  scores.reserve(edges_.size());
  const std::string mode = normalized_threshold_mode(options_);
  const bool use_local_scale = threshold_mode_uses_local_scale(mode);
  saddle_cut_activated_ = false;
  saddle_cut_base_clusters_ = 0;
  saddle_cut_cross_retained_fraction_ =
      std::numeric_limits<double>::quiet_NaN();
  manifold_filter_activated_ = false;
  manifold_filter_base_clusters_ = 0;
  manifold_filter_mean_anisotropy_ =
      std::numeric_limits<double>::quiet_NaN();
  manifold_filter_retained_edge_count_ = 0;

  for (auto &edge : edges_) {
    double score = edge.length;
    if (use_local_scale) {
      const double su = std::max(local_scale_[edge.u], kMinPadding);
      const double sv = std::max(local_scale_[edge.v], kMinPadding);
      score = edge.length / std::sqrt(su * sv);
    }
    edge.normalized_score = score;
    scores.push_back(score);
  }

  stats_.edge_count = edges_.size();
  stats_.retained_edge_count = 0;
  if (mode == "bucket_percentile" && options_.explicit_threshold <= 0.0 &&
      !edges_.empty()) {
    struct LocalGrid {
      std::size_t rows = 0;
      std::size_t cols = 0;
      double min_x = 0.0;
      double min_y = 0.0;
      double step_x = 1.0;
      double step_y = 1.0;
    };

    std::vector<const PointRecord *> active;
    active.reserve(records_.size());
    for (const auto &record : records_) {
      if (record.active)
        active.push_back(&record);
    }

    LocalGrid grid;
    grid.rows = grid.cols = desired_bucket_dimension(active.size());
    if (!active.empty()) {
      double min_x = active.front()->x, max_x = active.front()->x;
      double min_y = active.front()->y, max_y = active.front()->y;
      for (const auto *p : active) {
        min_x = std::min(min_x, p->x);
        max_x = std::max(max_x, p->x);
        min_y = std::min(min_y, p->y);
        max_y = std::max(max_y, p->y);
      }
      const double range_x = std::max(max_x - min_x, kMinPadding);
      const double range_y = std::max(max_y - min_y, kMinPadding);
      const double pad_x = std::max(range_x * kPaddingFraction, kMinPadding);
      const double pad_y = std::max(range_y * kPaddingFraction, kMinPadding);
      grid.min_x = min_x - pad_x;
      grid.min_y = min_y - pad_y;
      grid.step_x =
          std::max((range_x + 2.0 * pad_x) / static_cast<double>(grid.cols),
                   kMinPadding);
      grid.step_y =
          std::max((range_y + 2.0 * pad_y) / static_cast<double>(grid.rows),
                   kMinPadding);
    }

    auto cell_for = [&grid](double x, double y) {
      if (grid.rows == 0 || grid.cols == 0)
        return -1;
      const int col = std::max(
          0, std::min(static_cast<int>(grid.cols) - 1,
                      static_cast<int>(std::floor((x - grid.min_x) /
                                                  grid.step_x))));
      const int row = std::max(
          0, std::min(static_cast<int>(grid.rows) - 1,
                      static_cast<int>(std::floor((y - grid.min_y) /
                                                  grid.step_y))));
      return row * static_cast<int>(grid.cols) + col;
    };

    std::unordered_map<std::size_t, const PointRecord *> by_id;
    for (const auto *p : active)
      by_id[p->id] = p;

    std::vector<std::vector<double>> scores_by_cell(grid.rows * grid.cols);
    std::vector<int> edge_cells(edges_.size(), -1);
    for (std::size_t i = 0; i < edges_.size(); ++i) {
      const auto u = by_id.find(edges_[i].u);
      const auto v = by_id.find(edges_[i].v);
      if (u == by_id.end() || v == by_id.end())
        continue;
      const double mx = 0.5 * (u->second->x + v->second->x);
      const double my = 0.5 * (u->second->y + v->second->y);
      const int cell = cell_for(mx, my);
      edge_cells[i] = cell;
      if (cell >= 0)
        scores_by_cell[static_cast<std::size_t>(cell)].push_back(
            edges_[i].normalized_score);
    }

    std::vector<double> sorted_global = scores;
    std::sort(sorted_global.begin(), sorted_global.end());
    const double fallback_threshold =
        select_automatic_threshold(sorted_global);
    const std::vector<ThresholdTraceRecord> global_trace = threshold_trace_;
    const double percentile =
        std::min(0.95, std::max(0.35, options_.bucket_threshold_percentile));
    const std::size_t min_local_scores =
        static_cast<std::size_t>(std::max(
            8, options_.bucket_min_scores > 0
                   ? options_.bucket_min_scores
                   : std::max(16, 4 * default_min_cluster_size())));
    const int max_radius =
        std::max(1, options_.bucket_max_radius > 0
                        ? options_.bucket_max_radius
                        : 4);

    std::vector<double> edge_local_thresholds(edges_.size(),
                                              fallback_threshold);
    for (std::size_t i = 0; i < edges_.size(); ++i) {
      std::vector<double> local_scores;
      const int cell = edge_cells[i];
      if (cell >= 0 && grid.rows > 0 && grid.cols > 0) {
        const int row = cell / static_cast<int>(grid.cols);
        const int col = cell % static_cast<int>(grid.cols);
        for (int radius = 0; radius <= max_radius; ++radius) {
          for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
              if (std::max(std::abs(dr), std::abs(dc)) != radius)
                continue;
              const int rr = row + dr;
              const int cc = col + dc;
              if (rr < 0 || cc < 0 || rr >= static_cast<int>(grid.rows) ||
                  cc >= static_cast<int>(grid.cols)) {
                continue;
              }
              const std::size_t nb =
                  static_cast<std::size_t>(rr) * grid.cols +
                  static_cast<std::size_t>(cc);
              local_scores.insert(local_scores.end(), scores_by_cell[nb].begin(),
                                  scores_by_cell[nb].end());
            }
          }
          if (local_scores.size() >= min_local_scores)
            break;
        }
      }
      if (local_scores.size() < min_local_scores) {
        local_scores = sorted_global;
      } else {
        std::sort(local_scores.begin(), local_scores.end());
      }
      const double local_threshold =
          local_scores.empty()
              ? fallback_threshold
              : percentile_sorted(local_scores, percentile);
      edge_local_thresholds[i] = std::min(local_threshold, fallback_threshold);
    }

    std::vector<double> bucket_candidates;
    bucket_candidates.reserve(global_trace.size() + 1);
    for (const auto &row : global_trace) {
      bucket_candidates.push_back(std::min(row.threshold, fallback_threshold));
    }
    bucket_candidates.push_back(fallback_threshold);
    std::sort(bucket_candidates.begin(), bucket_candidates.end());
    bucket_candidates.erase(
        std::unique(bucket_candidates.begin(), bucket_candidates.end(),
                    [](double a, double b) { return std::abs(a - b) <= 1e-9; }),
        bucket_candidates.end());

    const int min_cluster_size =
        options_.min_cluster_size > 0 ? options_.min_cluster_size
                                      : default_min_cluster_size();
    const std::size_t active_count = count_active_records(records_);
    std::vector<ThresholdEvaluation> evaluated;
    std::vector<bool> nontrivial_flags;
    evaluated.reserve(bucket_candidates.size());
    nontrivial_flags.reserve(bucket_candidates.size());
    for (double candidate : bucket_candidates) {
      ThresholdEvaluation eval = evaluate_threshold_with_caps(
          records_, edges_, &edge_local_thresholds, candidate, min_cluster_size);
      const double noise_fraction =
          active_count == 0
              ? 0.0
              : static_cast<double>(eval.noise_count) /
                    static_cast<double>(active_count);
      const bool nontrivial = eval.num_clusters >= 2 && noise_fraction <= 0.35;
      evaluated.push_back(eval);
      nontrivial_flags.push_back(nontrivial);
    }
    apply_stability_objective_adjustment(evaluated, active_count,
                                         edges_.size());

    double best_base_objective = -std::numeric_limits<double>::infinity();
    double best_nontrivial_base_objective =
        -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < evaluated.size(); ++i) {
      best_base_objective =
          std::max(best_base_objective, evaluated[i].base_objective);
      if (nontrivial_flags[i]) {
        best_nontrivial_base_objective =
            std::max(best_nontrivial_base_objective,
                     evaluated[i].base_objective);
      }
    }

    ThresholdEvaluation best_objective;
    ThresholdEvaluation best_nontrivial;
    bool have_objective = false;
    bool have_nontrivial = false;
    const double stability_tie_band =
        std::max(0.0, options_.stability_tie_band);
    for (std::size_t i = 0; i < evaluated.size(); ++i) {
      const auto &eval = evaluated[i];
      const bool base_eligible =
          eval.base_objective + stability_tie_band >= best_base_objective;
      if (base_eligible &&
          (!have_objective || eval.objective > best_objective.objective ||
           (std::abs(eval.objective - best_objective.objective) <= 1e-9 &&
            eval.threshold > best_objective.threshold))) {
        best_objective = eval;
        have_objective = true;
      }
      const bool nontrivial_base_eligible =
          eval.base_objective + stability_tie_band >=
          best_nontrivial_base_objective;
      if (nontrivial_flags[i] && nontrivial_base_eligible &&
          (!have_nontrivial || eval.objective > best_nontrivial.objective ||
           (std::abs(eval.objective - best_nontrivial.objective) <= 1e-9 &&
            eval.threshold < best_nontrivial.threshold))) {
        best_nontrivial = eval;
        have_nontrivial = true;
      }
    }

    const double selected_threshold =
        have_nontrivial
            ? best_nontrivial.threshold
            : (have_objective ? best_objective.threshold : fallback_threshold);

    double threshold_sum = 0.0;
    std::size_t threshold_count = 0;
    for (std::size_t i = 0; i < edges_.size(); ++i) {
      const double capped_threshold =
          std::min(edge_local_thresholds[i], selected_threshold);
      edges_[i].threshold_cap = edge_local_thresholds[i];
      edges_[i].effective_threshold = capped_threshold;
      edges_[i].retained = edges_[i].normalized_score <= capped_threshold;
      if (edges_[i].retained)
        stats_.retained_edge_count++;
      threshold_sum += capped_threshold;
      threshold_count++;
    }
    stats_.threshold =
        threshold_count == 0 ? fallback_threshold
                             : threshold_sum /
                                   static_cast<double>(threshold_count);
    threshold_trace_.clear();
    threshold_trace_.reserve(evaluated.size());
    for (std::size_t i = 0; i < evaluated.size(); ++i) {
      const bool selected =
          std::abs(evaluated[i].threshold - selected_threshold) <= 1e-9;
      threshold_trace_.push_back(make_threshold_trace_record(
          evaluated[i], i, selected, nontrivial_flags[i], active_count,
          edges_.size()));
    }
    return;
  }

  if (options_.explicit_threshold > 0.0) {
    stats_.threshold = options_.explicit_threshold;
    record_explicit_threshold_trace();
  } else {
    stats_.threshold = select_automatic_threshold(scores);
  }

  for (auto &edge : edges_) {
    edge.threshold_cap = stats_.threshold;
    edge.effective_threshold = stats_.threshold;
    edge.retained = edge.normalized_score <= stats_.threshold;
    if (edge.retained)
      stats_.retained_edge_count++;
  }
  if (mode == "supported_merge")
    apply_supported_component_merge();
  else if (mode == "saddle_cut")
    apply_saddle_cut();
  else if (mode == "manifold_filter")
    apply_manifold_filter();
}

void DelaunayClusterer::apply_supported_component_merge() {
  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  if (records_.empty() || edges_.empty() || min_cluster_size <= 0)
    return;

  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  for (const auto &record : records_) {
    if (record.active)
      adj[record.id] = {};
  }

  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    auto u = adj.find(edge.u);
    auto v = adj.find(edge.v);
    if (u == adj.end() || v == adj.end())
      continue;
    u->second.push_back(edge.v);
    v->second.push_back(edge.u);
  }

  std::unordered_map<std::size_t, int> component_by_id;
  std::vector<std::vector<std::size_t>> components;
  std::unordered_set<std::size_t> visited;
  for (const auto &record : records_) {
    if (!record.active || visited.find(record.id) != visited.end())
      continue;

    std::vector<std::size_t> component;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component.push_back(curr);
      for (std::size_t nb : adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }

    if (static_cast<int>(component.size()) < min_cluster_size)
      continue;
    const int component_id = static_cast<int>(components.size());
    for (std::size_t id : component)
      component_by_id[id] = component_id;
    components.push_back(std::move(component));
  }

  if (components.size() < 2)
    return;

  const double factor = std::max(1.0, options_.supported_merge_factor);
  const std::size_t min_supported_edges = static_cast<std::size_t>(
      std::max(1, options_.supported_merge_min_edges));
  std::map<std::pair<int, int>, std::vector<std::size_t>> candidates;
  const double base_threshold =
      std::isfinite(stats_.threshold) ? stats_.threshold : 0.0;
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    const EdgeRecord &edge = edges_[i];
    if (edge.retained)
      continue;
    auto u = component_by_id.find(edge.u);
    auto v = component_by_id.find(edge.v);
    if (u == component_by_id.end() || v == component_by_id.end() ||
        u->second == v->second)
      continue;
    const double merge_threshold =
        std::max(base_threshold, edge.effective_threshold) * factor;
    if (edge.normalized_score > merge_threshold)
      continue;
    int a = u->second;
    int b = v->second;
    if (a > b)
      std::swap(a, b);
    candidates[{a, b}].push_back(i);
  }

  DisjointSet dsu(components.size());
  for (const auto &kv : candidates) {
    if (kv.second.size() >= min_supported_edges)
      dsu.unite(kv.first.first, kv.first.second);
  }

  std::size_t retained_count = stats_.retained_edge_count;
  for (const auto &kv : candidates) {
    if (dsu.find(kv.first.first) != dsu.find(kv.first.second))
      continue;
    for (std::size_t edge_index : kv.second) {
      EdgeRecord &edge = edges_[edge_index];
      if (edge.retained)
        continue;
      edge.retained = true;
      edge.effective_threshold =
          std::max(edge.effective_threshold, edge.normalized_score);
      edge.threshold_cap = std::max(edge.threshold_cap, edge.effective_threshold);
      retained_count++;
    }
  }
  stats_.retained_edge_count = retained_count;

  if (!threshold_trace_.empty()) {
    const ThresholdEvaluation final_eval = evaluate_current_retained(
        records_, edges_, stats_.threshold, min_cluster_size);
    for (std::size_t i = 0; i < threshold_trace_.size(); ++i) {
      if (!threshold_trace_[i].selected)
        continue;
      threshold_trace_[i] = make_threshold_trace_record(
          final_eval, i, true,
          final_eval.num_clusters >= 2 &&
              static_cast<double>(final_eval.noise_count) /
                      static_cast<double>(
                          std::max<std::size_t>(1, stats_.point_count)) <=
                  0.35,
          stats_.point_count, edges_.size());
      break;
    }
  }
}

void DelaunayClusterer::apply_saddle_cut() {
  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  if (records_.empty() || edges_.empty() || min_cluster_size <= 0)
    return;

  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  for (const auto &record : records_) {
    if (record.active)
      adj[record.id] = {};
  }
  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    auto u = adj.find(edge.u);
    auto v = adj.find(edge.v);
    if (u == adj.end() || v == adj.end())
      continue;
    u->second.push_back(edge.v);
    v->second.push_back(edge.u);
  }

  std::unordered_set<std::size_t> visited;
  std::size_t large_components = 0;
  for (const auto &record : records_) {
    if (!record.active || visited.find(record.id) != visited.end())
      continue;
    std::size_t component_size = 0;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component_size++;
      for (std::size_t nb : adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }
    if (static_cast<int>(component_size) >= min_cluster_size)
      large_components++;
  }

  saddle_cut_base_clusters_ = large_components;
  const std::size_t min_components = static_cast<std::size_t>(
      std::max(2, options_.saddle_cut_min_clusters));
  if (large_components < min_components)
    return;

  std::vector<const PointRecord *> active;
  active.reserve(records_.size());
  for (const auto &record : records_) {
    if (record.active)
      active.push_back(&record);
  }
  if (active.size() < 2)
    return;

  double mean_x = 0.0;
  double mean_y = 0.0;
  for (const auto *record : active) {
    mean_x += record->x;
    mean_y += record->y;
  }
  mean_x /= static_cast<double>(active.size());
  mean_y /= static_cast<double>(active.size());

  double cov_xx = 0.0;
  double cov_xy = 0.0;
  double cov_yy = 0.0;
  for (const auto *record : active) {
    const double dx = record->x - mean_x;
    const double dy = record->y - mean_y;
    cov_xx += dx * dx;
    cov_xy += dx * dy;
    cov_yy += dy * dy;
  }
  const double denom =
      active.size() > 1 ? static_cast<double>(active.size() - 1) : 1.0;
  cov_xx /= denom;
  cov_xy /= denom;
  cov_yy /= denom;

  const double trace = cov_xx + cov_yy;
  const double delta =
      std::sqrt((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy);
  const double lambda = 0.5 * (trace + delta);
  double axis_x = cov_xy;
  double axis_y = lambda - cov_xx;
  if (std::abs(axis_x) + std::abs(axis_y) <= kMinPadding) {
    axis_x = lambda - cov_yy;
    axis_y = cov_xy;
  }
  if (std::abs(axis_x) + std::abs(axis_y) <= kMinPadding) {
    axis_x = 1.0;
    axis_y = 0.0;
  }
  const double norm = std::sqrt(axis_x * axis_x + axis_y * axis_y);
  axis_x /= std::max(norm, kMinPadding);
  axis_y /= std::max(norm, kMinPadding);

  std::vector<double> projections;
  projections.reserve(active.size());
  std::unordered_map<std::size_t, double> projection_by_id;
  for (const auto *record : active) {
    const double projection =
        (record->x - mean_x) * axis_x + (record->y - mean_y) * axis_y;
    projections.push_back(projection);
    projection_by_id[record->id] = projection;
  }
  const std::size_t median_index = projections.size() / 2;
  std::nth_element(projections.begin(), projections.begin() + median_index,
                   projections.end());
  const double cut_projection = projections[median_index];

  std::unordered_map<std::size_t, bool> side_by_id;
  side_by_id.reserve(active.size());
  std::size_t left_count = 0;
  for (const auto *record : active) {
    const bool left = projection_by_id[record->id] <= cut_projection;
    side_by_id[record->id] = left;
    if (left)
      left_count++;
  }
  const std::size_t right_count = active.size() - left_count;
  if (left_count < static_cast<std::size_t>(min_cluster_size) ||
      right_count < static_cast<std::size_t>(min_cluster_size)) {
    return;
  }

  std::size_t retained_edges = 0;
  std::size_t retained_cross_edges = 0;
  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    auto u = side_by_id.find(edge.u);
    auto v = side_by_id.find(edge.v);
    if (u == side_by_id.end() || v == side_by_id.end())
      continue;
    retained_edges++;
    if (u->second != v->second)
      retained_cross_edges++;
  }
  const double cross_retained_fraction =
      retained_edges == 0
          ? 1.0
          : static_cast<double>(retained_cross_edges) /
                static_cast<double>(retained_edges);
  saddle_cut_cross_retained_fraction_ = cross_retained_fraction;
  if (cross_retained_fraction >
      std::max(0.0, options_.saddle_cut_max_cross_retained)) {
    return;
  }
  saddle_cut_activated_ = true;

  const double base_threshold =
      std::isfinite(stats_.threshold) ? stats_.threshold : 0.0;
  const double same_side_threshold =
      base_threshold * std::max(1.0, options_.saddle_cut_relax_factor);
  const double cross_threshold =
      base_threshold * std::max(0.0, options_.saddle_cut_cross_factor);

  stats_.retained_edge_count = 0;
  double threshold_sum = 0.0;
  std::size_t threshold_count = 0;
  for (auto &edge : edges_) {
    auto u = side_by_id.find(edge.u);
    auto v = side_by_id.find(edge.v);
    if (u == side_by_id.end() || v == side_by_id.end())
      continue;
    const bool same_side = u->second == v->second;
    const double effective_threshold =
        same_side ? same_side_threshold : cross_threshold;
    edge.threshold_cap = effective_threshold;
    edge.effective_threshold = effective_threshold;
    edge.retained = edge.normalized_score <= effective_threshold;
    if (edge.retained)
      stats_.retained_edge_count++;
    threshold_sum += effective_threshold;
    threshold_count++;
  }
  stats_.threshold = threshold_count == 0
                         ? base_threshold
                         : threshold_sum / static_cast<double>(threshold_count);

  if (!threshold_trace_.empty()) {
    const ThresholdEvaluation final_eval = evaluate_current_retained(
        records_, edges_, stats_.threshold, min_cluster_size);
    for (std::size_t i = 0; i < threshold_trace_.size(); ++i) {
      if (!threshold_trace_[i].selected)
        continue;
      threshold_trace_[i] = make_threshold_trace_record(
          final_eval, i, true,
          final_eval.num_clusters >= 2 &&
              static_cast<double>(final_eval.noise_count) /
                      static_cast<double>(
                          std::max<std::size_t>(1, stats_.point_count)) <=
                  0.35,
          stats_.point_count, edges_.size());
      break;
    }
  }
}

void DelaunayClusterer::apply_manifold_filter() {
  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  if (records_.empty() || edges_.empty() || min_cluster_size <= 0)
    return;

  std::vector<const PointRecord *> active;
  active.reserve(records_.size());
  std::unordered_map<std::size_t, const PointRecord *> by_id;
  by_id.reserve(records_.size());
  for (const auto &record : records_) {
    if (!record.active)
      continue;
    active.push_back(&record);
    by_id[record.id] = &record;
  }
  if (active.size() < 2)
    return;

  std::unordered_map<std::size_t, std::vector<std::size_t>> all_adj;
  std::unordered_map<std::size_t, std::vector<std::size_t>> retained_adj;
  all_adj.reserve(active.size());
  retained_adj.reserve(active.size());
  for (const auto *record : active) {
    all_adj[record->id] = {};
    retained_adj[record->id] = {};
  }
  for (const auto &edge : edges_) {
    const auto u = by_id.find(edge.u);
    const auto v = by_id.find(edge.v);
    if (u == by_id.end() || v == by_id.end())
      continue;
    all_adj[edge.u].push_back(edge.v);
    all_adj[edge.v].push_back(edge.u);
    if (!edge.retained)
      continue;
    retained_adj[edge.u].push_back(edge.v);
    retained_adj[edge.v].push_back(edge.u);
  }

  std::unordered_set<std::size_t> visited;
  std::size_t large_components = 0;
  for (const auto *record : active) {
    if (visited.find(record->id) != visited.end())
      continue;
    std::size_t component_size = 0;
    std::queue<std::size_t> q;
    q.push(record->id);
    visited.insert(record->id);
    while (!q.empty()) {
      const std::size_t curr = q.front();
      q.pop();
      component_size++;
      for (std::size_t nb : retained_adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }
    if (static_cast<int>(component_size) >= min_cluster_size)
      large_components++;
  }

  manifold_filter_base_clusters_ = large_components;
  const std::size_t min_components = static_cast<std::size_t>(
      std::max(2, options_.manifold_min_clusters));
  if (large_components < min_components)
    return;

  struct TangentStats {
    double tx = 1.0;
    double ty = 0.0;
    double anisotropy = 0.0;
    bool valid = false;
  };

  const std::size_t local_k = static_cast<std::size_t>(
      std::max(4, options_.knn_scale_k > 0 ? options_.knn_scale_k
                                           : default_min_cluster_size()));
  const std::size_t bfs_limit = std::max<std::size_t>(local_k * 3, local_k + 4);
  std::unordered_map<std::size_t, TangentStats> tangent_by_id;
  tangent_by_id.reserve(active.size());
  double anisotropy_sum = 0.0;
  std::size_t valid_tangents = 0;

  for (const auto *center : active) {
    std::vector<std::size_t> candidates;
    candidates.reserve(bfs_limit);
    std::unordered_set<std::size_t> local_seen;
    std::queue<std::size_t> q;
    local_seen.insert(center->id);
    q.push(center->id);
    while (!q.empty() && candidates.size() < bfs_limit) {
      const std::size_t curr = q.front();
      q.pop();
      const auto adj_it = all_adj.find(curr);
      if (adj_it == all_adj.end())
        continue;
      for (std::size_t nb : adj_it->second) {
        if (!local_seen.insert(nb).second)
          continue;
        candidates.push_back(nb);
        q.push(nb);
        if (candidates.size() >= bfs_limit)
          break;
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [&](std::size_t a, std::size_t b) {
                const auto *pa = by_id[a];
                const auto *pb = by_id[b];
                const double dax = pa->x - center->x;
                const double day = pa->y - center->y;
                const double dbx = pb->x - center->x;
                const double dby = pb->y - center->y;
                const double da = dax * dax + day * day;
                const double db = dbx * dbx + dby * dby;
                if (std::abs(da - db) > 1e-18)
                  return da < db;
                return a < b;
              });
    if (candidates.size() > local_k)
      candidates.resize(local_k);
    if (candidates.size() < 2) {
      tangent_by_id[center->id] = {};
      continue;
    }

    double mean_x = center->x;
    double mean_y = center->y;
    std::size_t count = 1;
    for (std::size_t id : candidates) {
      const auto *p = by_id[id];
      mean_x += p->x;
      mean_y += p->y;
      count++;
    }
    mean_x /= static_cast<double>(count);
    mean_y /= static_cast<double>(count);

    double cov_xx = (center->x - mean_x) * (center->x - mean_x);
    double cov_xy = (center->x - mean_x) * (center->y - mean_y);
    double cov_yy = (center->y - mean_y) * (center->y - mean_y);
    for (std::size_t id : candidates) {
      const auto *p = by_id[id];
      const double dx = p->x - mean_x;
      const double dy = p->y - mean_y;
      cov_xx += dx * dx;
      cov_xy += dx * dy;
      cov_yy += dy * dy;
    }
    const double denom =
        count > 1 ? static_cast<double>(count - 1) : 1.0;
    cov_xx /= denom;
    cov_xy /= denom;
    cov_yy /= denom;

    const double trace = cov_xx + cov_yy;
    const double delta = std::sqrt((cov_xx - cov_yy) * (cov_xx - cov_yy) +
                                   4.0 * cov_xy * cov_xy);
    if (trace <= kMinPadding) {
      tangent_by_id[center->id] = {};
      continue;
    }

    const double lambda = 0.5 * (trace + delta);
    double axis_x = cov_xy;
    double axis_y = lambda - cov_xx;
    if (std::abs(axis_x) + std::abs(axis_y) <= kMinPadding) {
      axis_x = lambda - cov_yy;
      axis_y = cov_xy;
    }
    if (std::abs(axis_x) + std::abs(axis_y) <= kMinPadding) {
      axis_x = 1.0;
      axis_y = 0.0;
    }
    const double norm = std::sqrt(axis_x * axis_x + axis_y * axis_y);
    TangentStats stats;
    stats.tx = axis_x / std::max(norm, kMinPadding);
    stats.ty = axis_y / std::max(norm, kMinPadding);
    stats.anisotropy = std::min(1.0, std::max(0.0, delta / trace));
    stats.valid = true;
    tangent_by_id[center->id] = stats;
    anisotropy_sum += stats.anisotropy;
    valid_tangents++;
  }

  if (valid_tangents == 0)
    return;
  manifold_filter_mean_anisotropy_ =
      anisotropy_sum / static_cast<double>(valid_tangents);
  if (manifold_filter_mean_anisotropy_ <
      std::max(0.0, options_.manifold_dataset_anisotropy_min)) {
    return;
  }

  const double base_threshold =
      std::isfinite(stats_.threshold) ? stats_.threshold : 0.0;
  if (base_threshold <= 0.0)
    return;

  const double core_threshold =
      base_threshold * std::max(0.0, options_.manifold_core_factor);
  const double bridge_threshold =
      base_threshold *
      std::max(options_.manifold_core_factor, options_.manifold_bridge_factor);
  const double min_alignment =
      std::min(1.0, std::max(0.0, options_.manifold_alignment_min));
  const double min_anisotropy =
      std::min(1.0, std::max(0.0, options_.manifold_anisotropy_min));

  stats_.retained_edge_count = 0;
  double threshold_sum = 0.0;
  std::size_t threshold_count = 0;
  for (auto &edge : edges_) {
    const auto u_record = by_id.find(edge.u);
    const auto v_record = by_id.find(edge.v);
    if (u_record == by_id.end() || v_record == by_id.end())
      continue;

    const auto u_tangent = tangent_by_id.find(edge.u);
    const auto v_tangent = tangent_by_id.find(edge.v);
    bool manifold_edge = false;
    if (u_tangent != tangent_by_id.end() && v_tangent != tangent_by_id.end() &&
        u_tangent->second.valid && v_tangent->second.valid) {
      const double dx = v_record->second->x - u_record->second->x;
      const double dy = v_record->second->y - u_record->second->y;
      const double length = std::sqrt(dx * dx + dy * dy);
      if (length > kMinPadding) {
        const double ux = dx / length;
        const double uy = dy / length;
        const double align_u =
            std::abs(ux * u_tangent->second.tx + uy * u_tangent->second.ty);
        const double align_v =
            std::abs(ux * v_tangent->second.tx + uy * v_tangent->second.ty);
        const double alignment = std::min(align_u, align_v);
        const double anisotropy = std::min(u_tangent->second.anisotropy,
                                           v_tangent->second.anisotropy);
        manifold_edge = edge.normalized_score <= bridge_threshold &&
                        alignment >= min_alignment &&
                        anisotropy >= min_anisotropy;
      }
    }

    const bool core_edge = edge.normalized_score <= core_threshold;
    const double effective_threshold =
        manifold_edge ? bridge_threshold : core_threshold;
    edge.threshold_cap = effective_threshold;
    edge.effective_threshold = effective_threshold;
    edge.retained = core_edge || manifold_edge;
    if (edge.retained)
      stats_.retained_edge_count++;
    threshold_sum += effective_threshold;
    threshold_count++;
  }

  stats_.threshold = threshold_count == 0
                         ? base_threshold
                         : threshold_sum / static_cast<double>(threshold_count);
  manifold_filter_activated_ = true;
  manifold_filter_retained_edge_count_ = stats_.retained_edge_count;

  if (!threshold_trace_.empty()) {
    const ThresholdEvaluation final_eval = evaluate_current_retained(
        records_, edges_, stats_.threshold, min_cluster_size);
    const std::size_t active_count = active.size();
    for (std::size_t i = 0; i < threshold_trace_.size(); ++i) {
      if (!threshold_trace_[i].selected)
        continue;
      threshold_trace_[i] = make_threshold_trace_record(
          final_eval, i, true,
          final_eval.num_clusters >= 2 &&
              static_cast<double>(final_eval.noise_count) /
                      static_cast<double>(std::max<std::size_t>(
                          1, active_count)) <=
                  0.35,
          active_count, edges_.size());
      break;
    }
  }
}

double DelaunayClusterer::select_automatic_threshold(
    const std::vector<double> &scores) {
  if (scores.empty())
    return std::numeric_limits<double>::infinity();

  std::vector<double> sorted = scores;
  std::sort(sorted.begin(), sorted.end());
  const double upper =
      robust_threshold(sorted, options_.threshold_iqr_multiplier);

  std::vector<double> candidates;
  for (double p = 0.35; p <= 0.90; p += 0.025) {
    const double candidate = percentile_sorted(sorted, p);
    if (candidate <= upper || candidates.empty())
      candidates.push_back(candidate);
  }
  candidates.push_back(upper);
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end(),
                               [](double a, double b) {
                                 return std::abs(a - b) <= 1e-9;
                               }),
                   candidates.end());

  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  const std::size_t active_count = count_active_records(records_);

  std::vector<ThresholdEvaluation> evaluated;
  std::vector<bool> nontrivial_flags;
  evaluated.reserve(candidates.size());
  nontrivial_flags.reserve(candidates.size());
  for (double candidate : candidates) {
    ThresholdEvaluation eval =
        evaluate_threshold(records_, edges_, candidate, min_cluster_size);
    const double noise_fraction =
        active_count == 0
            ? 0.0
            : static_cast<double>(eval.noise_count) /
                  static_cast<double>(active_count);
    const bool nontrivial = eval.num_clusters >= 2 && noise_fraction <= 0.35;
    evaluated.push_back(eval);
    nontrivial_flags.push_back(nontrivial);
  }
  apply_stability_objective_adjustment(evaluated, active_count, edges_.size());

  double best_base_objective = -std::numeric_limits<double>::infinity();
  double best_nontrivial_base_objective =
      -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < evaluated.size(); ++i) {
    best_base_objective =
        std::max(best_base_objective, evaluated[i].base_objective);
    if (nontrivial_flags[i]) {
      best_nontrivial_base_objective =
          std::max(best_nontrivial_base_objective,
                   evaluated[i].base_objective);
    }
  }

  ThresholdEvaluation best_objective;
  ThresholdEvaluation best_nontrivial;
  bool have_objective = false;
  bool have_nontrivial = false;
  const double stability_tie_band = std::max(0.0, options_.stability_tie_band);
  for (std::size_t i = 0; i < evaluated.size(); ++i) {
    const auto &eval = evaluated[i];
    const bool base_eligible =
        eval.base_objective + stability_tie_band >= best_base_objective;
    if (base_eligible &&
        (!have_objective || eval.objective > best_objective.objective ||
         (std::abs(eval.objective - best_objective.objective) <= 1e-9 &&
          eval.threshold > best_objective.threshold))) {
      best_objective = eval;
      have_objective = true;
    }
    const bool nontrivial_base_eligible =
        eval.base_objective + stability_tie_band >=
        best_nontrivial_base_objective;
    if (nontrivial_flags[i] && nontrivial_base_eligible &&
        (!have_nontrivial || eval.objective > best_nontrivial.objective ||
         (std::abs(eval.objective - best_nontrivial.objective) <= 1e-9 &&
          eval.threshold < best_nontrivial.threshold))) {
      best_nontrivial = eval;
      have_nontrivial = true;
    }
  }

  const double selected_threshold =
      have_nontrivial
          ? best_nontrivial.threshold
          : (have_objective ? best_objective.threshold : upper);
  threshold_trace_.clear();
  threshold_trace_.reserve(evaluated.size());
  for (std::size_t i = 0; i < evaluated.size(); ++i) {
    const bool selected =
        std::abs(evaluated[i].threshold - selected_threshold) <= 1e-9;
    threshold_trace_.push_back(make_threshold_trace_record(
        evaluated[i], i, selected, nontrivial_flags[i], active_count,
        edges_.size()));
  }
  return selected_threshold;
}

void DelaunayClusterer::record_explicit_threshold_trace() {
  threshold_trace_.clear();
  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  const std::size_t active_count = count_active_records(records_);
  ThresholdEvaluation eval =
      evaluate_threshold(records_, edges_, stats_.threshold, min_cluster_size);
  const double noise_fraction =
      active_count == 0
          ? 0.0
          : static_cast<double>(eval.noise_count) /
                static_cast<double>(active_count);
  const bool nontrivial = eval.num_clusters >= 2 && noise_fraction <= 0.35;
  threshold_trace_.push_back(make_threshold_trace_record(
      eval, 0, true, nontrivial, active_count, edges_.size()));
}

int DelaunayClusterer::default_min_cluster_size() const {
  const std::size_t n = active_ids().size();
  if (n <= 2)
    return 1;
  return std::max(1, static_cast<int>(std::ceil(std::log2(n))));
}

std::vector<std::size_t> DelaunayClusterer::active_ids() const {
  std::vector<std::size_t> ids;
  for (const auto &record : records_) {
    if (record.active)
      ids.push_back(record.id);
  }
  return ids;
}

std::vector<int> DelaunayClusterer::affected_cells_for_positions(
    const std::vector<std::pair<double, double>> &positions) const {
  std::set<int> cells;
  for (const auto &pos : positions) {
    const int center = bucket_index_.bucket_index(pos.first, pos.second);
    for (int idx : bucket_index_.cell_neighborhood(center, 1))
      cells.insert(idx);
  }
  return std::vector<int>(cells.begin(), cells.end());
}

void DelaunayClusterer::populate_bucket_impact(
    DynamicUpdateReport &report,
    const std::vector<std::pair<double, double>> &positions) const {
  const auto affected_cells = affected_cells_for_positions(positions);
  std::unordered_set<std::size_t> point_ids;
  std::unordered_set<std::size_t> edge_ids;
  const auto &cells = bucket_index_.cells();
  for (int idx : affected_cells) {
    if (idx < 0 || static_cast<std::size_t>(idx) >= cells.size())
      continue;
    const auto &cell = cells[static_cast<std::size_t>(idx)];
    point_ids.insert(cell.point_ids.begin(), cell.point_ids.end());
    edge_ids.insert(cell.edge_ids.begin(), cell.edge_ids.end());
  }
  report.affected_cells = affected_cells.size();
  report.affected_bucket_points = point_ids.size();
  report.affected_bucket_edges = edge_ids.size();
}

void DelaunayClusterer::rebuild_edge_lookup() {
  edge_index_by_id_.clear();
  edge_ids_by_vertex_.clear();
  for (std::size_t i = 0; i < edges_.size(); ++i)
    register_edge_record(i);
}

void DelaunayClusterer::register_edge_record(std::size_t edge_index) {
  if (edge_index >= edges_.size())
    return;
  const EdgeRecord &edge = edges_[edge_index];
  edge_index_by_id_[edge.id] = edge_index;
  edge_ids_by_vertex_[edge.u].push_back(edge.id);
  edge_ids_by_vertex_[edge.v].push_back(edge.id);
}

bool DelaunayClusterer::erase_edge_by_id(std::size_t edge_id) {
  auto idx_it = edge_index_by_id_.find(edge_id);
  if (idx_it == edge_index_by_id_.end())
    return false;

  const std::size_t idx = idx_it->second;
  const EdgeRecord removed = edges_[idx];
  edge_index_by_id_.erase(idx_it);

  auto remove_incident = [&](std::size_t vertex_id) {
    auto inc = edge_ids_by_vertex_.find(vertex_id);
    if (inc == edge_ids_by_vertex_.end())
      return;
    erase_value(inc->second, edge_id);
    if (inc->second.empty())
      edge_ids_by_vertex_.erase(inc);
  };
  remove_incident(removed.u);
  remove_incident(removed.v);

  const std::size_t last = edges_.size() - 1;
  if (idx != last) {
    edges_[idx] = edges_[last];
    edge_index_by_id_[edges_[idx].id] = idx;
  }
  edges_.pop_back();
  return true;
}

std::set<std::pair<std::size_t, std::size_t>>
DelaunayClusterer::retained_edge_keys() const {
  std::set<std::pair<std::size_t, std::size_t>> keys;
  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    std::size_t a = edge.u;
    std::size_t b = edge.v;
    if (a > b)
      std::swap(a, b);
    keys.insert({a, b});
  }
  return keys;
}

bool DelaunayClusterer::refresh_edges_scales_and_pruning_locally(
    const std::set<std::size_t> &seed_ids, std::size_t *scope_vertices,
    std::size_t *candidate_edges) {
  last_local_edge_refresh_tracked_ = false;
  last_bucket_removed_edge_ids_.clear();
  last_bucket_upsert_edge_ids_.clear();
  last_removed_retained_edges_.clear();
  last_added_retained_edges_.clear();

	  if (options_.explicit_threshold <= 0.0)
	    return false;
	  if (!threshold_mode_supports_local_edge_refresh(options_))
	    return false;
	  const std::string mode = normalized_threshold_mode(options_);
	  const bool use_local_scale = threshold_mode_uses_local_scale(mode);

  std::set<std::size_t> scope = seed_ids;
  for (int depth = 0; depth < 2; ++depth) {
    std::set<std::size_t> next = scope;
    for (std::size_t id : scope) {
      auto old_edge_it = edge_ids_by_vertex_.find(id);
      if (old_edge_it != edge_ids_by_vertex_.end()) {
        for (std::size_t edge_id : old_edge_it->second) {
          auto edge_idx = edge_index_by_id_.find(edge_id);
          if (edge_idx == edge_index_by_id_.end())
            continue;
          const EdgeRecord &edge = edges_[edge_idx->second];
          next.insert(edge.u == id ? edge.v : edge.u);
        }
      }
      auto handle = vertex_handles_.find(id);
      if (handle == vertex_handles_.end())
        continue;
      auto circ = dt_.incident_vertices(handle->second);
      auto start = circ;
      if (circ != nullptr) {
        do {
          if (!dt_.is_infinite(circ))
            next.insert(circ->info());
          ++circ;
        } while (circ != start);
      }
    }
    scope.swap(next);
  }

  std::set<std::size_t> active_scope;
  for (std::size_t id : scope) {
    auto idx = id_to_index_.find(id);
    if (idx != id_to_index_.end() && records_[idx->second].active)
      active_scope.insert(id);
  }
  if (active_scope.empty() && stats_.point_count > 0)
    return false;

  auto make_key = [](std::size_t a, std::size_t b) {
    if (a > b)
      std::swap(a, b);
    return std::make_pair(a, b);
  };

  auto is_active_id = [this](std::size_t id) {
    auto idx = id_to_index_.find(id);
    return idx != id_to_index_.end() && records_[idx->second].active;
  };

  std::map<std::pair<std::size_t, std::size_t>, std::size_t> old_id_by_key;
  std::set<std::size_t> touched_old_edge_ids;
  std::set<std::pair<std::size_t, std::size_t>> touched_keys;
  std::set<std::pair<std::size_t, std::size_t>> old_retained_touched_keys;
  std::set<std::pair<std::size_t, std::size_t>> current_retained_touched_keys;
  for (std::size_t id : scope) {
    auto inc = edge_ids_by_vertex_.find(id);
    if (inc == edge_ids_by_vertex_.end())
      continue;
    touched_old_edge_ids.insert(inc->second.begin(), inc->second.end());
  }

  std::size_t retained_count = stats_.retained_edge_count;
  for (std::size_t edge_id : touched_old_edge_ids) {
    auto edge_idx = edge_index_by_id_.find(edge_id);
    if (edge_idx == edge_index_by_id_.end())
      continue;
    const EdgeRecord edge = edges_[edge_idx->second];
    const auto key = make_key(edge.u, edge.v);
    old_id_by_key[key] = edge.id;
    touched_keys.insert(key);
    last_bucket_removed_edge_ids_.insert(edge.id);
    if (edge.retained) {
      old_retained_touched_keys.insert(key);
    }
    if (edge.retained && retained_count > 0)
      retained_count--;
    erase_edge_by_id(edge.id);
  }

  std::vector<EdgeRecord> current_edges;
  std::set<std::pair<std::size_t, std::size_t>> current_keys;
  auto collect_current_edge = [&](std::size_t a, std::size_t b) {
    if (a == b || !is_active_id(a) || !is_active_id(b))
      return;
    const auto key = make_key(a, b);
    if (!current_keys.insert(key).second)
      return;
    touched_keys.insert(key);
    const auto &ra = records_[id_to_index_.at(a)];
    const auto &rb = records_[id_to_index_.at(b)];
    EdgeRecord edge;
    auto old_id = old_id_by_key.find(key);
    edge.id =
        old_id == old_id_by_key.end() ? next_edge_id_++ : old_id->second;
    edge.u = key.first;
    edge.v = key.second;
    edge.length = distance_xy(ra.x, ra.y, rb.x, rb.y);
    last_bucket_upsert_edge_ids_.insert(edge.id);
    current_edges.push_back(edge);
  };

  for (std::size_t id : active_scope) {
    auto handle = vertex_handles_.find(id);
    if (handle == vertex_handles_.end())
      continue;
    auto circ = dt_.incident_vertices(handle->second);
    auto start = circ;
    if (circ != nullptr) {
      do {
        if (!dt_.is_infinite(circ))
          collect_current_edge(id, circ->info());
        ++circ;
      } while (circ != start);
    }
  }

  std::set<std::size_t> scale_scope = active_scope;
  for (const auto &key : touched_keys) {
    if (is_active_id(key.first))
      scale_scope.insert(key.first);
    if (is_active_id(key.second))
      scale_scope.insert(key.second);
  }

  for (std::size_t id : scope) {
    if (!is_active_id(id))
      local_scale_.erase(id);
  }

  for (std::size_t id : scale_scope) {
    std::vector<double> lengths;
    auto handle = vertex_handles_.find(id);
    if (handle != vertex_handles_.end()) {
      auto circ = dt_.incident_vertices(handle->second);
      auto start = circ;
      if (circ != nullptr) {
        do {
          if (!dt_.is_infinite(circ)) {
            const std::size_t nb = circ->info();
            const auto &ra = records_[id_to_index_.at(id)];
            const auto &rb = records_[id_to_index_.at(nb)];
            lengths.push_back(distance_xy(ra.x, ra.y, rb.x, rb.y));
          }
          ++circ;
        } while (circ != start);
      }
    }
    if (!lengths.empty()) {
      std::sort(lengths.begin(), lengths.end());
      local_scale_[id] = std::max(median_sorted(lengths), kMinPadding);
    } else if (local_scale_.find(id) == local_scale_.end()) {
      local_scale_[id] = kMinPadding;
    }
  }

  for (EdgeRecord &edge : current_edges) {
    edge.retained = false;
    edge.normalized_score = 0.0;
    edge.threshold_cap = options_.explicit_threshold;
    edge.effective_threshold = options_.explicit_threshold;
    edges_.push_back(edge);
    register_edge_record(edges_.size() - 1);
  }

  std::set<std::size_t> rescore_edge_ids;
  for (std::size_t id : scale_scope) {
    auto inc = edge_ids_by_vertex_.find(id);
    if (inc != edge_ids_by_vertex_.end())
      rescore_edge_ids.insert(inc->second.begin(), inc->second.end());
  }

  for (std::size_t edge_id : rescore_edge_ids) {
    auto edge_idx = edge_index_by_id_.find(edge_id);
    if (edge_idx == edge_index_by_id_.end())
      continue;
    EdgeRecord &edge = edges_[edge_idx->second];
    const bool was_retained = edge.retained;
    double score = edge.length;
	    if (use_local_scale) {
	      const double su = std::max(local_scale_[edge.u], kMinPadding);
	      const double sv = std::max(local_scale_[edge.v], kMinPadding);
	      score = edge.length / std::sqrt(su * sv);
    }
    edge.normalized_score = score;
    edge.threshold_cap = options_.explicit_threshold;
    edge.effective_threshold = options_.explicit_threshold;
    edge.retained = edge.normalized_score <= options_.explicit_threshold;
    const auto key = make_key(edge.u, edge.v);
    if (edge.retained && !was_retained) {
      retained_count++;
    } else if (!edge.retained && was_retained && retained_count > 0) {
      retained_count--;
    }
    if (touched_keys.find(key) != touched_keys.end()) {
      if (edge.retained)
        current_retained_touched_keys.insert(key);
    } else if (edge.retained != was_retained) {
      if (edge.retained)
        last_added_retained_edges_.insert(key);
      else
        last_removed_retained_edges_.insert(key);
    }
  }

  std::set_difference(old_retained_touched_keys.begin(),
                      old_retained_touched_keys.end(),
                      current_retained_touched_keys.begin(),
                      current_retained_touched_keys.end(),
                      std::inserter(last_removed_retained_edges_,
                                    last_removed_retained_edges_.end()));
  std::set_difference(current_retained_touched_keys.begin(),
                      current_retained_touched_keys.end(),
                      old_retained_touched_keys.begin(),
                      old_retained_touched_keys.end(),
                      std::inserter(last_added_retained_edges_,
                                    last_added_retained_edges_.end()));

  stats_.point_count = dt_.number_of_vertices();
  stats_.threshold = options_.explicit_threshold;
  stats_.edge_count = edges_.size();
  stats_.retained_edge_count = retained_count;

  if (scope_vertices)
    *scope_vertices = scale_scope.size();
  if (candidate_edges)
    *candidate_edges = touched_keys.size();
  last_local_edge_refresh_tracked_ = true;
  return true;
}

std::size_t DelaunayClusterer::verify_retained_edges_against_full_recompute()
    const {
  std::vector<PointRecord> active;
  active.reserve(records_.size());
  for (const auto &record : records_) {
    if (record.active) {
      PointRecord copy = record;
      copy.cluster_id = -1;
      copy.is_noise = false;
      copy.active = true;
      active.push_back(copy);
    }
  }

  DelaunayClusterer scratch(options_);
  ClusterOptions scratch_options = options_;
  scratch_options.experimental_incremental_geometry = false;
  scratch_options.experimental_local_edge_refresh = false;
  scratch_options.experimental_local_bucket_refresh = false;
  scratch_options.experimental_local_relabel = false;
  scratch.set_options(scratch_options);
  scratch.fit(active);

  auto current = retained_edge_keys();
  auto full = scratch.retained_edge_keys();
  std::vector<std::pair<std::size_t, std::size_t>> diff;
  std::set_symmetric_difference(current.begin(), current.end(), full.begin(),
                                full.end(), std::back_inserter(diff));
  return diff.size();
}

std::size_t DelaunayClusterer::verify_bucket_index_against_frozen_rebuild()
    const {
  std::size_t active_count = 0;
  for (const auto &record : records_) {
    if (record.active)
      active_count++;
  }
  if (active_count > 0 &&
      (bucket_index_.rows() == 0 || bucket_index_.cols() == 0))
    return active_count;

  SpatialBucketIndex expected = bucket_index_;
  expected.rebuild_on_current_layout(records_, edges_);

  std::size_t mismatches = 0;

  std::set<std::size_t> point_ids;
  for (const auto &kv : bucket_index_.point_to_cell())
    point_ids.insert(kv.first);
  for (const auto &kv : expected.point_to_cell())
    point_ids.insert(kv.first);
  for (std::size_t id : point_ids) {
    auto current = bucket_index_.point_to_cell().find(id);
    auto rebuilt = expected.point_to_cell().find(id);
    if (current == bucket_index_.point_to_cell().end() ||
        rebuilt == expected.point_to_cell().end() ||
        current->second != rebuilt->second) {
      mismatches++;
    }
  }

  std::set<std::size_t> edge_ids;
  for (const auto &kv : bucket_index_.edge_to_cells())
    edge_ids.insert(kv.first);
  for (const auto &kv : expected.edge_to_cells())
    edge_ids.insert(kv.first);
  for (std::size_t id : edge_ids) {
    auto current = bucket_index_.edge_to_cells().find(id);
    auto rebuilt = expected.edge_to_cells().find(id);
    if (current == bucket_index_.edge_to_cells().end() ||
        rebuilt == expected.edge_to_cells().end()) {
      mismatches++;
      continue;
    }
    if (sorted_copy(current->second) != sorted_copy(rebuilt->second))
      mismatches++;
  }

  const std::size_t cell_count =
      std::max(bucket_index_.cells().size(), expected.cells().size());
  for (std::size_t i = 0; i < cell_count; ++i) {
    const BucketCell empty;
    const BucketCell &current =
        i < bucket_index_.cells().size() ? bucket_index_.cells()[i] : empty;
    const BucketCell &rebuilt =
        i < expected.cells().size() ? expected.cells()[i] : empty;
    if (sorted_copy(current.point_ids) != sorted_copy(rebuilt.point_ids))
      mismatches++;
    if (sorted_copy(current.edge_ids) != sorted_copy(rebuilt.edge_ids))
      mismatches++;
  }

  return mismatches;
}

bool DelaunayClusterer::retained_labeling_has_split() const {
  // Sound check that the current labeling equals the connected-components +
  // noise labeling of the retained graph (the partition a from-scratch fit would
  // produce). We rebuild components by BFS and verify three invariants:
  //   (1) every member of a component shares one cluster id (no under-merge / partial
  //       relabel),
  //   (2) is_noise iff component size < min_cluster_size (no noise-threshold drift),
  //   (3) each non-noise cluster id maps to exactly one component (no over-merge).
  // Any violation means the incremental relabel diverged and a full relabel is needed.
  // BFS on compact retained adjacency is expected O(n+r). Constructing it here
  // also scans N stored records (including inactive ones) and e Delaunay edges:
  // expected O(N+e+n+r), or O(N+n) in 2D. N may grow with session history.
  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  for (const auto &record : records_)
    if (record.active)
      adj[record.id];
  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    if (adj.count(edge.u) && adj.count(edge.v)) {
      adj[edge.u].push_back(edge.v);
      adj[edge.v].push_back(edge.u);
    }
  }
  const int min_cluster_size = options_.min_cluster_size > 0
                                   ? options_.min_cluster_size
                                   : default_min_cluster_size();
  std::unordered_set<std::size_t> visited;
  std::unordered_map<int, std::size_t> cluster_owner;
  for (const auto &record : records_) {
    if (!record.active || visited.count(record.id))
      continue;
    std::vector<std::size_t> comp;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      const std::size_t c = q.front();
      q.pop();
      comp.push_back(c);
      for (std::size_t nb : adj[c])
        if (visited.insert(nb).second)
          q.push(nb);
    }
    const bool should_noise = static_cast<int>(comp.size()) < min_cluster_size;
    const int rep_cluster = records_[id_to_index_.at(comp[0])].cluster_id;
    for (std::size_t id : comp) {
      const PointRecord &p = records_[id_to_index_.at(id)];
      if (p.is_noise != should_noise)
        return true;
      if (!should_noise && p.cluster_id != rep_cluster)
        return true;
    }
    if (!should_noise) {
      if (!cluster_owner.emplace(rep_cluster, comp[0]).second)
        return true;
    }
  }
  return false;
}

bool DelaunayClusterer::repair_labeling_if_inconsistent(
    std::size_t *changed_labels) {
  // Correctness guard for the incremental relabel: after a local edit the
  // heuristic labeling may diverge from a from-scratch fit, so we check it with a
  // global scan (see retained_labeling_has_split for stored-record costs), which
  // rebuilds the retained adjacency,
  // BFS-enumerates components, and re-evaluates the min-cluster-size noise
  // threshold, ceil(log2 n) by default, from the live active count. On any
  // inconsistency we fall back to a full relabel; the from-scratch verifier
  // (--verify-dynamic) independently confirms exactness either way.
  // Ablation only: --no-guard skips the check entirely (heuristic-only mode);
  // updates are then NOT guaranteed exact.
  if (options_.disable_labeling_guard)
    return false;
  // --always-full-relabel labels canonically by construction; the guard is
  // redundant there.
  if (options_.always_full_relabel)
    return false;
  if (!retained_labeling_has_split())
    return false;
  label_connected_components(changed_labels);
  return true;
}

void DelaunayClusterer::label_connected_components(std::size_t *changed_labels) {
  if (changed_labels)
    *changed_labels = 0;

  std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
  for (const auto &record : records_) {
    if (record.active)
      adj[record.id] = {};
  }
  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    adj[edge.u].push_back(edge.v);
    adj[edge.v].push_back(edge.u);
  }

  const int min_cluster_size =
      options_.min_cluster_size > 0 ? options_.min_cluster_size
                                    : default_min_cluster_size();
  std::unordered_set<std::size_t> visited;
  int next_cluster = 0;
  std::size_t noise = 0;
  cluster_sizes_.clear();
  cluster_members_.clear();

  for (auto &record : records_) {
    if (!record.active)
      continue;
    if (visited.find(record.id) != visited.end())
      continue;

    std::vector<std::size_t> component;
    std::queue<std::size_t> q;
    q.push(record.id);
    visited.insert(record.id);
    while (!q.empty()) {
      std::size_t curr = q.front();
      q.pop();
      component.push_back(curr);
      for (std::size_t nb : adj[curr]) {
        if (visited.insert(nb).second)
          q.push(nb);
      }
    }

    const bool is_noise = static_cast<int>(component.size()) < min_cluster_size;
    const int cluster_id = is_noise ? -1 : next_cluster++;
    if (!is_noise) {
      cluster_sizes_[cluster_id] = component.size();
      cluster_members_[cluster_id].insert(component.begin(), component.end());
    }
    for (std::size_t id : component) {
      PointRecord &p = records_[id_to_index_.at(id)];
      if (changed_labels &&
          (p.cluster_id != cluster_id || p.is_noise != is_noise)) {
        (*changed_labels)++;
      }
      p.cluster_id = cluster_id;
      p.is_noise = is_noise;
      if (is_noise)
        noise++;
    }
  }

  stats_.num_clusters = static_cast<std::size_t>(next_cluster);
  stats_.noise_count = noise;
  if (connectivity_summary_enabled(options_)) {
    refresh_retained_connectivity_summary();
  } else {
    retained_bridge_edges_.clear();
    retained_articulation_vertices_.clear();
    retained_bridge_edges_by_cluster_.clear();
    retained_articulation_vertices_by_cluster_.clear();
    retained_connectivity_summary_clusters_.clear();
  }
}

void DelaunayClusterer::refresh_cluster_stats_from_labels(
    bool refresh_connectivity_summary) {
  std::set<int> clusters;
  std::size_t noise = 0;
  cluster_sizes_.clear();
  cluster_members_.clear();
  for (const auto &record : records_) {
    if (!record.active)
      continue;
    if (record.is_noise || record.cluster_id < 0) {
      noise++;
    } else {
      clusters.insert(record.cluster_id);
      cluster_sizes_[record.cluster_id]++;
      cluster_members_[record.cluster_id].insert(record.id);
    }
  }
  stats_.num_clusters = clusters.size();
  stats_.noise_count = noise;
  if (refresh_connectivity_summary && connectivity_summary_enabled(options_)) {
    refresh_retained_connectivity_summary();
  } else if (refresh_connectivity_summary) {
    retained_bridge_edges_.clear();
    retained_articulation_vertices_.clear();
    retained_bridge_edges_by_cluster_.clear();
    retained_articulation_vertices_by_cluster_.clear();
    retained_connectivity_summary_clusters_.clear();
  }
}

void DelaunayClusterer::refresh_retained_connectivity_summary() {
  retained_bridge_edges_.clear();
  retained_articulation_vertices_.clear();
  retained_bridge_edges_by_cluster_.clear();
  retained_articulation_vertices_by_cluster_.clear();
  retained_connectivity_summary_clusters_.clear();

  std::map<int, std::unordered_map<std::size_t, std::vector<std::size_t>>>
      adj_by_cluster;
  for (const auto &record : records_) {
    if (!record.active || record.is_noise || record.cluster_id < 0)
      continue;
    auto size_it = cluster_sizes_.find(record.cluster_id);
    if (size_it == cluster_sizes_.end() ||
        size_it->second > kMaxConnectivitySummaryClusterSize)
      continue;
    adj_by_cluster[record.cluster_id][record.id] = {};
  }

  for (const auto &edge : edges_) {
    if (!edge.retained)
      continue;
    auto u_idx = id_to_index_.find(edge.u);
    auto v_idx = id_to_index_.find(edge.v);
    if (u_idx == id_to_index_.end() || v_idx == id_to_index_.end())
      continue;
    const PointRecord &u = records_[u_idx->second];
    const PointRecord &v = records_[v_idx->second];
    if (!u.active || !v.active || u.is_noise || v.is_noise ||
        u.cluster_id < 0 || u.cluster_id != v.cluster_id)
      continue;
    auto cluster_it = adj_by_cluster.find(u.cluster_id);
    if (cluster_it == adj_by_cluster.end())
      continue;
    auto u_adj = cluster_it->second.find(edge.u);
    auto v_adj = cluster_it->second.find(edge.v);
    if (u_adj == cluster_it->second.end() ||
        v_adj == cluster_it->second.end())
      continue;
    u_adj->second.push_back(edge.v);
    v_adj->second.push_back(edge.u);
  }

  for (auto &kv : adj_by_cluster) {
    std::set<std::pair<std::size_t, std::size_t>> cluster_bridges;
    std::set<std::size_t> cluster_articulations;
    bridge_articulation_summary(kv.second, cluster_bridges,
                                cluster_articulations);
    retained_connectivity_summary_clusters_.insert(kv.first);
    if (!cluster_bridges.empty()) {
      retained_bridge_edges_by_cluster_[kv.first] = cluster_bridges;
      retained_bridge_edges_.insert(cluster_bridges.begin(),
                                    cluster_bridges.end());
    }
    if (!cluster_articulations.empty()) {
      retained_articulation_vertices_by_cluster_[kv.first] =
          cluster_articulations;
      retained_articulation_vertices_.insert(cluster_articulations.begin(),
                                             cluster_articulations.end());
    }
  }
}

void DelaunayClusterer::refresh_retained_connectivity_summary_for_clusters(
    const std::set<int> &cluster_ids) {
  for (int cluster_id : cluster_ids) {
    auto bridge_it = retained_bridge_edges_by_cluster_.find(cluster_id);
    if (bridge_it != retained_bridge_edges_by_cluster_.end()) {
      for (const auto &edge : bridge_it->second)
        retained_bridge_edges_.erase(edge);
      retained_bridge_edges_by_cluster_.erase(bridge_it);
    }

    auto articulation_it =
        retained_articulation_vertices_by_cluster_.find(cluster_id);
    if (articulation_it != retained_articulation_vertices_by_cluster_.end()) {
      for (std::size_t id : articulation_it->second)
        retained_articulation_vertices_.erase(id);
      retained_articulation_vertices_by_cluster_.erase(articulation_it);
    }
    retained_connectivity_summary_clusters_.erase(cluster_id);

    std::unordered_map<std::size_t, std::vector<std::size_t>> adj;
    auto members_it = cluster_members_.find(cluster_id);
    if (members_it == cluster_members_.end())
      continue;
    if (members_it->second.size() > kMaxConnectivitySummaryClusterSize)
      continue;

    for (std::size_t id : members_it->second) {
      auto idx = id_to_index_.find(id);
      if (idx == id_to_index_.end())
        continue;
      const PointRecord &record = records_[idx->second];
      if (!record.active || record.is_noise || record.cluster_id != cluster_id)
        continue;
      adj[id] = {};
    }
    if (adj.empty())
      continue;

    std::set<std::pair<std::size_t, std::size_t>> seen_edges;
    for (const auto &kv : adj) {
      auto inc = edge_ids_by_vertex_.find(kv.first);
      if (inc == edge_ids_by_vertex_.end())
        continue;
      for (std::size_t edge_id : inc->second) {
        auto edge_idx = edge_index_by_id_.find(edge_id);
        if (edge_idx == edge_index_by_id_.end())
          continue;
        const EdgeRecord &edge = edges_[edge_idx->second];
        if (!edge.retained)
          continue;
        const auto key = ordered_pair(edge.u, edge.v);
        if (!seen_edges.insert(key).second)
          continue;
        auto u_it = adj.find(edge.u);
        auto v_it = adj.find(edge.v);
        if (u_it == adj.end() || v_it == adj.end())
          continue;
        u_it->second.push_back(edge.v);
        v_it->second.push_back(edge.u);
      }
    }

    std::set<std::pair<std::size_t, std::size_t>> cluster_bridges;
    std::set<std::size_t> cluster_articulations;
    bridge_articulation_summary(adj, cluster_bridges, cluster_articulations);
    retained_connectivity_summary_clusters_.insert(cluster_id);

    if (!cluster_bridges.empty()) {
      retained_bridge_edges_by_cluster_[cluster_id] = cluster_bridges;
      retained_bridge_edges_.insert(cluster_bridges.begin(),
                                    cluster_bridges.end());
    }
    if (!cluster_articulations.empty()) {
      retained_articulation_vertices_by_cluster_[cluster_id] =
          cluster_articulations;
      retained_articulation_vertices_.insert(cluster_articulations.begin(),
                                             cluster_articulations.end());
    }
  }
}

std::size_t DelaunayClusterer::nearest_active_id(double x, double y) const {
  if (dt_.number_of_vertices() > 0) {
    VertexHandle v = dt_.nearest_vertex(Point(x, y));
    if (v != VertexHandle())
      return v->info();
  }

  double best = std::numeric_limits<double>::infinity();
  std::size_t best_id = 0;
  bool found = false;
  for (const auto &record : records_) {
    if (!record.active)
      continue;
    const double d = sqr(record.x - x) + sqr(record.y - y);
    if (d < best) {
      best = d;
      best_id = record.id;
      found = true;
    }
  }
  if (!found)
    throw std::runtime_error("No active points in clusterer");
  return best_id;
}

std::size_t DelaunayClusterer::estimate_affected_vertices(double x,
                                                          double y) const {
  if (dt_.number_of_vertices() == 0)
    return 0;
  std::unordered_set<std::size_t> affected;
  VertexHandle v = dt_.nearest_vertex(Point(x, y));
  if (v == VertexHandle())
    return 0;
  affected.insert(v->info());
  auto circ = dt_.incident_vertices(v);
  auto start = circ;
  if (circ != nullptr) {
    do {
      if (!dt_.is_infinite(circ))
        affected.insert(circ->info());
      ++circ;
    } while (circ != start);
  }
  return affected.size();
}

std::size_t DelaunayClusterer::estimate_affected_edges(double x, double y) const {
  std::size_t affected_vertices = estimate_affected_vertices(x, y);
  if (affected_vertices == 0)
    return 0;
  return std::min(edges_.size(), affected_vertices * 6);
}

DynamicUpdateReport DelaunayClusterer::insert_point(double x, double y,
                                                    int label,
                                                    bool has_label) {
  DynamicUpdateReport report;
  report.operation = "insert";
  report.incremental_geometry_used = options_.experimental_incremental_geometry;
  report.local_edge_refresh_used = options_.experimental_local_edge_refresh;
  report.local_bucket_refresh_used = options_.experimental_local_bucket_refresh;
  report.local_relabel_used = options_.experimental_local_relabel;
  const auto start = std::chrono::high_resolution_clock::now();
  const bool defer_bucket_refresh = options_.experimental_local_bucket_refresh;

  // Delaunay clustering assumes general position: each location is a distinct
  // vertex. An insert coinciding exactly with an existing vertex would alias two
  // ids onto one CGAL vertex -- corrupting the handle map (a crash on a later
  // delete) and breaking exactness against a from-scratch fit. The location is
  // already represented, so reject the duplicate as a no-op before mutating any
  // state. Exact predicates make this coincidence test exact.
  if (options_.experimental_incremental_geometry && dt_.number_of_vertices() > 0) {
    Delaunay::Locate_type lt;
    int li;
    dt_.locate(Point(x, y), lt, li);
    if (lt == Delaunay::VERTEX) {
      report.incremental_geometry_used = false;
      const auto end = std::chrono::high_resolution_clock::now();
      report.time_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      return report;
    }
  }

  PointRecord rec;
  rec.id = next_id_++;
  rec.x = x;
  rec.y = y;
  rec.label = label;
  rec.has_label = has_label;
  rec.active = true;
  id_to_index_[rec.id] = records_.size();
  records_.push_back(rec);
  if (options_.experimental_incremental_geometry) {
    VertexHandle handle = dt_.insert(Point(x, y));
    handle->info() = rec.id;
    vertex_handles_[rec.id] = handle;
  }

  std::size_t changed = 0;
  std::size_t relabel_scope = 0;
  std::size_t relabel_search = 0;
  if (options_.experimental_local_relabel) {
    const std::set<std::size_t> active_added_ids{rec.id};
    const std::set<std::size_t> active_removed_ids;
    const std::unordered_map<std::size_t, std::vector<std::size_t>>
        removed_old_retained_neighbors;
    rebuild_model_with_local_relabel(
        {rec.id}, active_added_ids, active_removed_ids,
        removed_old_retained_neighbors, &changed, &relabel_scope,
        &relabel_search,
        &report.edge_refresh_scope_vertices, &report.edge_refresh_candidate_edges,
        &report.local_edge_refresh_fallback,
        options_.experimental_incremental_geometry, defer_bucket_refresh);
  } else if (options_.experimental_incremental_geometry) {
    refresh_model_from_current_triangulation(&changed, defer_bucket_refresh);
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  } else {
    if (defer_bucket_refresh) {
      rebuild_triangulation();
      refresh_model_from_current_triangulation(&changed, true);
    } else {
      rebuild_model(&changed);
    }
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  }
  if (options_.experimental_local_relabel) {
    std::size_t repair_changed = 0;
    if (repair_labeling_if_inconsistent(&repair_changed)) {
      report.local_relabel_fallback = true;
      changed += repair_changed;
      relabel_scope = stats_.point_count;
      relabel_search = stats_.point_count;
    }
  }
  if (defer_bucket_refresh)
    refresh_bucket_index_after_dynamic({rec.id}, report);
  populate_bucket_impact(report, {{x, y}});
  report.affected_vertices = estimate_affected_vertices(x, y);
  report.affected_edges = estimate_affected_edges(x, y);
  report.changed_labels = changed;
  report.relabel_scope_vertices = relabel_scope;
  report.relabel_search_vertices = relabel_search;
  report.retained_edges_removed = last_relabel_removed_retained_edges_count_;
  report.retained_edges_added = last_relabel_added_retained_edges_count_;
  report.small_side_split_used = last_small_side_split_used_;
  report.point_delete_split_used = last_point_delete_split_used_;

  const auto end = std::chrono::high_resolution_clock::now();
  report.time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return report;
}

DynamicUpdateReport DelaunayClusterer::remove_nearest(double x, double y) {
  DynamicUpdateReport report;
  report.operation = "delete";
  report.incremental_geometry_used = options_.experimental_incremental_geometry;
  report.local_edge_refresh_used = options_.experimental_local_edge_refresh;
  report.local_bucket_refresh_used = options_.experimental_local_bucket_refresh;
  report.local_relabel_used = options_.experimental_local_relabel;
  const auto start = std::chrono::high_resolution_clock::now();
  const bool defer_bucket_refresh = options_.experimental_local_bucket_refresh;

  const std::size_t affected_before = estimate_affected_vertices(x, y);
  const std::size_t edge_before = estimate_affected_edges(x, y);
  std::set<std::size_t> seed_ids;
  std::set<std::size_t> touched_point_ids;
  std::unordered_map<std::size_t, std::vector<std::size_t>>
      removed_old_retained_neighbors;
  if (stats_.point_count > 0) {
    const std::size_t id = nearest_active_id(x, y);
    touched_point_ids.insert(id);
    auto incident = edge_ids_by_vertex_.find(id);
    if (incident != edge_ids_by_vertex_.end()) {
      for (std::size_t edge_id : incident->second) {
        auto edge_idx = edge_index_by_id_.find(edge_id);
        if (edge_idx == edge_index_by_id_.end())
          continue;
        const EdgeRecord &edge = edges_[edge_idx->second];
        if (!edge.retained)
          continue;
        const std::size_t other = edge.u == id ? edge.v : edge.u;
        seed_ids.insert(other);
        removed_old_retained_neighbors[id].push_back(other);
      }
    }
    seed_ids.insert(id);
    if (options_.experimental_incremental_geometry) {
      auto handle = vertex_handles_.find(id);
      if (handle != vertex_handles_.end()) {
        dt_.remove(handle->second);
        vertex_handles_.erase(handle);
      } else {
        report.incremental_geometry_used = false;
      }
    }
    records_[id_to_index_.at(id)].active = false;
  }

  std::size_t changed = 0;
  std::size_t relabel_scope = 0;
  std::size_t relabel_search = 0;
  if (options_.experimental_local_relabel) {
    rebuild_model_with_local_relabel(
        seed_ids, {}, touched_point_ids, removed_old_retained_neighbors,
        &changed, &relabel_scope, &relabel_search,
        &report.edge_refresh_scope_vertices,
        &report.edge_refresh_candidate_edges, &report.local_edge_refresh_fallback,
        report.incremental_geometry_used, defer_bucket_refresh);
  } else if (report.incremental_geometry_used) {
    refresh_model_from_current_triangulation(&changed, defer_bucket_refresh);
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  } else {
    if (defer_bucket_refresh) {
      rebuild_triangulation();
      refresh_model_from_current_triangulation(&changed, true);
    } else {
      rebuild_model(&changed);
    }
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  }
  if (options_.experimental_local_relabel) {
    std::size_t repair_changed = 0;
    if (repair_labeling_if_inconsistent(&repair_changed)) {
      report.local_relabel_fallback = true;
      changed += repair_changed;
      relabel_scope = stats_.point_count;
      relabel_search = stats_.point_count;
    }
  }
  if (defer_bucket_refresh)
    refresh_bucket_index_after_dynamic(touched_point_ids, report);
  populate_bucket_impact(report, {{x, y}});
  report.affected_vertices = affected_before;
  report.affected_edges = edge_before;
  report.changed_labels = changed;
  report.relabel_scope_vertices = relabel_scope;
  report.relabel_search_vertices = relabel_search;
  report.retained_edges_removed = last_relabel_removed_retained_edges_count_;
  report.retained_edges_added = last_relabel_added_retained_edges_count_;
  report.small_side_split_used = last_small_side_split_used_;
  report.point_delete_split_used = last_point_delete_split_used_;

  const auto end = std::chrono::high_resolution_clock::now();
  report.time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return report;
}

DynamicUpdateReport DelaunayClusterer::move_nearest(double old_x, double old_y,
                                                    double new_x,
                                                    double new_y) {
  DynamicUpdateReport report;
  report.operation = "move";
  report.incremental_geometry_used = options_.experimental_incremental_geometry;
  report.local_edge_refresh_used = options_.experimental_local_edge_refresh;
  report.local_bucket_refresh_used = options_.experimental_local_bucket_refresh;
  report.local_relabel_used = options_.experimental_local_relabel;
  const auto start = std::chrono::high_resolution_clock::now();
  const bool defer_bucket_refresh = options_.experimental_local_bucket_refresh;

  const std::size_t affected_before = estimate_affected_vertices(old_x, old_y);
  const std::size_t edge_before = estimate_affected_edges(old_x, old_y);
  std::set<std::size_t> seed_ids;
  std::set<std::size_t> touched_point_ids;
  if (stats_.point_count > 0) {
    const std::size_t id = nearest_active_id(old_x, old_y);
    // Reject a move whose target coincides exactly with a *different* existing
    // vertex: aliasing two ids onto one CGAL vertex would corrupt the handle map
    // and break exactness. Moving onto its own location is fine. (See the
    // general-position note in insert_point.)
    if (options_.experimental_incremental_geometry) {
      auto moving = vertex_handles_.find(id);
      Delaunay::Locate_type lt;
      int li;
      Delaunay::Face_handle f = dt_.locate(Point(new_x, new_y), lt, li);
      if (lt == Delaunay::VERTEX &&
          (moving == vertex_handles_.end() || f->vertex(li) != moving->second)) {
        const auto end = std::chrono::high_resolution_clock::now();
        report.time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        return report;
      }
    }
    touched_point_ids.insert(id);
    seed_ids.insert(id);
    auto incident = edge_ids_by_vertex_.find(id);
    if (incident != edge_ids_by_vertex_.end()) {
      for (std::size_t edge_id : incident->second) {
        auto edge_idx = edge_index_by_id_.find(edge_id);
        if (edge_idx == edge_index_by_id_.end())
          continue;
        const EdgeRecord &edge = edges_[edge_idx->second];
        if (!edge.retained)
          continue;
        seed_ids.insert(edge.u == id ? edge.v : edge.u);
      }
    }
    PointRecord &rec = records_[id_to_index_.at(id)];
    rec.x = new_x;
    rec.y = new_y;
    if (options_.experimental_incremental_geometry) {
      auto handle = vertex_handles_.find(id);
      if (handle != vertex_handles_.end()) {
        dt_.remove(handle->second);
        VertexHandle inserted = dt_.insert(Point(new_x, new_y));
        inserted->info() = id;
        handle->second = inserted;
      } else {
        report.incremental_geometry_used = false;
      }
    }
  }

  std::size_t changed = 0;
  std::size_t relabel_scope = 0;
  std::size_t relabel_search = 0;
  if (options_.experimental_local_relabel) {
    const std::set<std::size_t> active_added_ids;
    const std::set<std::size_t> active_removed_ids;
    const std::unordered_map<std::size_t, std::vector<std::size_t>>
        removed_old_retained_neighbors;
    rebuild_model_with_local_relabel(
        seed_ids, active_added_ids, active_removed_ids,
        removed_old_retained_neighbors, &changed, &relabel_scope,
        &relabel_search,
        &report.edge_refresh_scope_vertices, &report.edge_refresh_candidate_edges,
        &report.local_edge_refresh_fallback, report.incremental_geometry_used,
        defer_bucket_refresh);
  } else if (report.incremental_geometry_used) {
    refresh_model_from_current_triangulation(&changed, defer_bucket_refresh);
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  } else {
    if (defer_bucket_refresh) {
      rebuild_triangulation();
      refresh_model_from_current_triangulation(&changed, true);
    } else {
      rebuild_model(&changed);
    }
    relabel_scope = stats_.point_count;
    relabel_search = stats_.point_count;
  }
  if (options_.experimental_local_relabel) {
    std::size_t repair_changed = 0;
    if (repair_labeling_if_inconsistent(&repair_changed)) {
      report.local_relabel_fallback = true;
      changed += repair_changed;
      relabel_scope = stats_.point_count;
      relabel_search = stats_.point_count;
    }
  }
  if (defer_bucket_refresh)
    refresh_bucket_index_after_dynamic(touched_point_ids, report);
  populate_bucket_impact(report, {{old_x, old_y}, {new_x, new_y}});
  report.affected_vertices =
      affected_before + estimate_affected_vertices(new_x, new_y);
  report.affected_edges = edge_before + estimate_affected_edges(new_x, new_y);
  report.changed_labels = changed;
  report.relabel_scope_vertices = relabel_scope;
  report.relabel_search_vertices = relabel_search;
  report.retained_edges_removed = last_relabel_removed_retained_edges_count_;
  report.retained_edges_added = last_relabel_added_retained_edges_count_;
  report.small_side_split_used = last_small_side_split_used_;
  report.point_delete_split_used = last_point_delete_split_used_;

  const auto end = std::chrono::high_resolution_clock::now();
  report.time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return report;
}

int DelaunayClusterer::assign_cluster(double x, double y) const {
  if (records_.empty())
    return -1;
  const std::size_t id = nearest_active_id(x, y);
  return records_[id_to_index_.at(id)].cluster_id;
}

bool DelaunayClusterer::assign_is_noise(double x, double y) const {
  if (records_.empty())
    return true;
  const std::size_t id = nearest_active_id(x, y);
  return records_[id_to_index_.at(id)].is_noise;
}

void DelaunayClusterer::write_clusters_csv(const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write clusters CSV: " + path);
  out << "x,y,cluster_id,is_noise";
  bool any_label = false;
  for (const auto &p : records_)
    any_label = any_label || p.has_label;
  if (any_label)
    out << ",label";
  out << "\n";
  out << std::setprecision(17);
  for (const auto &p : records_) {
    if (!p.active)
      continue;
    out << p.x << "," << p.y << "," << p.cluster_id << ","
        << (p.is_noise ? 1 : 0);
    if (any_label)
      out << "," << (p.has_label ? p.label : -1);
    out << "\n";
  }
}

void DelaunayClusterer::write_edges_csv(const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write edges CSV: " + path);
  out << "edge_id,u,v,length,normalized_score,threshold_cap,"
         "effective_threshold,threshold_margin,retained\n";
  out << std::setprecision(17);
  for (const auto &e : edges_) {
    out << e.id << "," << e.u << "," << e.v << "," << e.length << ","
        << e.normalized_score << "," << e.threshold_cap << ","
        << e.effective_threshold << ","
        << (e.effective_threshold - e.normalized_score) << ","
        << (e.retained ? 1 : 0) << "\n";
  }
}

void DelaunayClusterer::write_threshold_trace_csv(
    const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write threshold trace CSV: " + path);
  out << "candidate_index,threshold,selected,nontrivial,active_points,"
         "edge_count,retained_edges,retained_edge_pct,num_clusters,"
         "noise_count,noise_pct,largest_cluster,largest_cluster_pct,"
         "component_entropy,base_objective,stability_adjustment,"
         "plateau_width_candidates,plateau_center_offset,"
         "largest_jump_before_pct,largest_jump_after_pct,"
         "noise_drop_after_pct,objective\n";
  out << std::setprecision(17);
  for (const auto &row : threshold_trace_) {
    out << row.candidate_index << "," << row.threshold << ","
        << (row.selected ? 1 : 0) << "," << (row.nontrivial ? 1 : 0)
        << "," << row.active_points << "," << row.edge_count << ","
        << row.retained_edges << "," << row.retained_edge_pct << ","
        << row.num_clusters << "," << row.noise_count << ","
        << row.noise_pct << "," << row.largest_cluster << ","
        << row.largest_cluster_pct << "," << row.component_entropy << ","
        << row.base_objective << "," << row.stability_adjustment << ","
        << row.plateau_width_candidates << "," << row.plateau_center_offset
        << "," << row.largest_jump_before_pct << ","
        << row.largest_jump_after_pct << "," << row.noise_drop_after_pct << ","
        << row.objective << "\n";
  }
}

void DelaunayClusterer::write_bucket_points_csv(const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write bucket point registry: " + path);
  out << "point_id,cell_id,row,col\n";
  std::vector<std::pair<std::size_t, int>> rows(
      bucket_index_.point_to_cell().begin(), bucket_index_.point_to_cell().end());
  std::sort(rows.begin(), rows.end());
  for (const auto &kv : rows) {
    out << kv.first << "," << kv.second << ","
        << bucket_index_.cell_row(kv.second) << ","
        << bucket_index_.cell_col(kv.second) << "\n";
  }
}

void DelaunayClusterer::write_bucket_edges_csv(const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write bucket edge registry: " + path);
  out << "edge_id,cell_id,row,col\n";
  std::vector<std::pair<std::size_t, std::vector<int>>> rows(
      bucket_index_.edge_to_cells().begin(), bucket_index_.edge_to_cells().end());
  std::sort(rows.begin(), rows.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  for (const auto &kv : rows) {
    for (int cell_id : kv.second) {
      out << kv.first << "," << cell_id << ","
          << bucket_index_.cell_row(cell_id) << ","
          << bucket_index_.cell_col(cell_id) << "\n";
    }
  }
}

std::size_t DelaunayClusterer::verify_against_full_recompute() const {
  std::vector<PointRecord> active;
  active.reserve(records_.size());
  for (const auto &record : records_) {
    if (record.active) {
      PointRecord copy = record;
      copy.cluster_id = -1;
      copy.is_noise = false;
      copy.active = true;
      active.push_back(copy);
    }
  }

  DelaunayClusterer scratch(options_);
  scratch.fit(active);
  return partition_mismatches(records_, scratch.records());
}

void DelaunayClusterer::write_metadata(const std::string &path) const {
  ensure_parent_dir(path);
  std::ofstream out(path);
  if (!out.is_open())
    throw std::runtime_error("Cannot write metadata: " + path);
  out << "point_count," << stats_.point_count << "\n";
  out << "edge_count," << stats_.edge_count << "\n";
  out << "retained_edge_count," << stats_.retained_edge_count << "\n";
  out << "num_clusters," << stats_.num_clusters << "\n";
  out << "noise_count," << stats_.noise_count << "\n";
  out << "threshold," << std::setprecision(17) << stats_.threshold << "\n";
  out << "threshold_mode," << normalized_threshold_mode(options_) << "\n";
  out << "threshold_trace_rows," << threshold_trace_.size() << "\n";
  out << "bucket_threshold_percentile," << std::setprecision(17)
      << options_.bucket_threshold_percentile << "\n";
  out << "bucket_min_scores,"
      << (options_.bucket_min_scores > 0
              ? options_.bucket_min_scores
              : std::max(16, 4 * default_min_cluster_size()))
      << "\n";
  out << "bucket_max_radius,"
      << (options_.bucket_max_radius > 0 ? options_.bucket_max_radius : 4)
      << "\n";
  out << "knn_scale_k,"
      << (options_.knn_scale_k > 0 ? options_.knn_scale_k : default_min_cluster_size())
      << "\n";
  out << "supported_merge_factor," << std::setprecision(17)
      << std::max(1.0, options_.supported_merge_factor) << "\n";
  out << "supported_merge_min_edges,"
      << std::max(1, options_.supported_merge_min_edges) << "\n";
  out << "saddle_cut_relax_factor," << std::setprecision(17)
      << std::max(1.0, options_.saddle_cut_relax_factor) << "\n";
  out << "saddle_cut_cross_factor," << std::setprecision(17)
      << std::max(0.0, options_.saddle_cut_cross_factor) << "\n";
  out << "saddle_cut_min_clusters,"
      << std::max(2, options_.saddle_cut_min_clusters) << "\n";
  out << "saddle_cut_max_cross_retained," << std::setprecision(17)
      << std::max(0.0, options_.saddle_cut_max_cross_retained) << "\n";
  out << "saddle_cut_activated," << (saddle_cut_activated_ ? 1 : 0)
      << "\n";
  out << "saddle_cut_base_clusters," << saddle_cut_base_clusters_ << "\n";
  out << "saddle_cut_cross_retained_fraction," << std::setprecision(17)
      << saddle_cut_cross_retained_fraction_ << "\n";
  out << "manifold_core_factor," << std::setprecision(17)
      << std::max(0.0, options_.manifold_core_factor) << "\n";
  out << "manifold_bridge_factor," << std::setprecision(17)
      << std::max(options_.manifold_core_factor,
                  options_.manifold_bridge_factor)
      << "\n";
  out << "manifold_alignment_min," << std::setprecision(17)
      << std::min(1.0, std::max(0.0, options_.manifold_alignment_min))
      << "\n";
  out << "manifold_anisotropy_min," << std::setprecision(17)
      << std::min(1.0, std::max(0.0, options_.manifold_anisotropy_min))
      << "\n";
  out << "manifold_dataset_anisotropy_min," << std::setprecision(17)
      << std::max(0.0, options_.manifold_dataset_anisotropy_min) << "\n";
  out << "manifold_min_clusters,"
      << std::max(2, options_.manifold_min_clusters) << "\n";
  out << "manifold_filter_activated,"
      << (manifold_filter_activated_ ? 1 : 0) << "\n";
  out << "manifold_filter_base_clusters,"
      << manifold_filter_base_clusters_ << "\n";
  out << "manifold_filter_mean_anisotropy," << std::setprecision(17)
      << manifold_filter_mean_anisotropy_ << "\n";
  out << "manifold_filter_retained_edge_count,"
      << manifold_filter_retained_edge_count_ << "\n";
  out << "stability_tie_band," << std::setprecision(17)
      << std::max(0.0, options_.stability_tie_band) << "\n";
  out << "min_cluster_size,"
      << (options_.min_cluster_size > 0 ? options_.min_cluster_size
                                        : default_min_cluster_size())
      << "\n";
  out << "bucket_rows," << stats_.bucket_rows << "\n";
  out << "bucket_cols," << stats_.bucket_cols << "\n";
  out << "bucket_nonempty_point_cells," << stats_.bucket_nonempty_point_cells
      << "\n";
  out << "bucket_nonempty_edge_cells," << stats_.bucket_nonempty_edge_cells
      << "\n";
  out << "bucket_max_points," << stats_.bucket_max_points << "\n";
  out << "bucket_max_edges," << stats_.bucket_max_edges << "\n";
  out << "bucket_mean_points_nonempty," << std::setprecision(17)
      << stats_.bucket_mean_points_nonempty << "\n";
  out << "bucket_mean_edges_nonempty," << std::setprecision(17)
      << stats_.bucket_mean_edges_nonempty << "\n";
  out << "global_threshold_ablation,"
      << (options_.global_threshold_ablation ? 1 : 0) << "\n";
  out << "experimental_local_relabel,"
      << (options_.experimental_local_relabel ? 1 : 0) << "\n";
  out << "experimental_incremental_geometry,"
      << (options_.experimental_incremental_geometry ? 1 : 0) << "\n";
  out << "experimental_local_edge_refresh,"
      << (options_.experimental_local_edge_refresh ? 1 : 0) << "\n";
  out << "experimental_local_bucket_refresh,"
      << (options_.experimental_local_bucket_refresh ? 1 : 0) << "\n";
  out << "experimental_component_witness,"
      << (options_.experimental_component_witness ? 1 : 0) << "\n";
  out << "experimental_connectivity_summary,"
      << (options_.experimental_connectivity_summary ? 1 : 0) << "\n";
  out << "experimental_small_side_split,"
      << (options_.experimental_small_side_split ? 1 : 0) << "\n";
}

void DelaunayClusterer::save_model(const std::string &dir) const {
  std::filesystem::create_directories(dir);
  write_clusters_csv(dir + "/clusters.csv");
  write_edges_csv(dir + "/edges.csv");
  write_threshold_trace_csv(dir + "/threshold_trace.csv");
  write_bucket_points_csv(dir + "/bucket_points.csv");
  write_bucket_edges_csv(dir + "/bucket_edges.csv");
  write_metadata(dir + "/metadata.csv");

  std::ofstream points(dir + "/points.csv");
  points << "x,y,label\n";
  points << std::setprecision(17);
  for (const auto &p : records_) {
    if (!p.active)
      continue;
    points << p.x << "," << p.y << "," << (p.has_label ? p.label : -1)
           << "\n";
  }
}

void DelaunayClusterer::load_model(const std::string &dir) {
  const std::string metadata_path = dir + "/metadata.csv";
  std::ifstream metadata(metadata_path);
  if (metadata.is_open()) {
    ClusterOptions loaded = options_;
    std::string line;
    while (std::getline(metadata, line)) {
	      auto parts = split_csv_line(line);
	      if (parts.size() < 2)
	        continue;
	      if (parts[0] == "threshold_mode") {
	        loaded.threshold_mode = parts[1];
	        continue;
	      }
	      double value = 0.0;
	      if (!parse_double(parts[1], value))
	        continue;
      if (parts[0] == "threshold") {
        loaded.explicit_threshold = value;
      } else if (parts[0] == "bucket_threshold_percentile") {
        loaded.bucket_threshold_percentile = value;
      } else if (parts[0] == "bucket_min_scores") {
        loaded.bucket_min_scores = static_cast<int>(std::llround(value));
      } else if (parts[0] == "bucket_max_radius") {
        loaded.bucket_max_radius = static_cast<int>(std::llround(value));
      } else if (parts[0] == "knn_scale_k") {
        loaded.knn_scale_k = static_cast<int>(std::llround(value));
      } else if (parts[0] == "supported_merge_factor") {
        loaded.supported_merge_factor = std::max(1.0, value);
      } else if (parts[0] == "supported_merge_min_edges") {
        loaded.supported_merge_min_edges =
            std::max(1, static_cast<int>(std::llround(value)));
      } else if (parts[0] == "saddle_cut_relax_factor") {
        loaded.saddle_cut_relax_factor = std::max(1.0, value);
      } else if (parts[0] == "saddle_cut_cross_factor") {
        loaded.saddle_cut_cross_factor = std::max(0.0, value);
      } else if (parts[0] == "saddle_cut_min_clusters") {
        loaded.saddle_cut_min_clusters =
            std::max(2, static_cast<int>(std::llround(value)));
      } else if (parts[0] == "saddle_cut_max_cross_retained") {
        loaded.saddle_cut_max_cross_retained = std::max(0.0, value);
      } else if (parts[0] == "manifold_core_factor") {
        loaded.manifold_core_factor = std::max(0.0, value);
      } else if (parts[0] == "manifold_bridge_factor") {
        loaded.manifold_bridge_factor =
            std::max(loaded.manifold_core_factor, value);
      } else if (parts[0] == "manifold_alignment_min") {
        loaded.manifold_alignment_min = std::min(1.0, std::max(0.0, value));
      } else if (parts[0] == "manifold_anisotropy_min") {
        loaded.manifold_anisotropy_min = std::min(1.0, std::max(0.0, value));
      } else if (parts[0] == "manifold_dataset_anisotropy_min") {
        loaded.manifold_dataset_anisotropy_min = std::max(0.0, value);
      } else if (parts[0] == "manifold_min_clusters") {
        loaded.manifold_min_clusters =
            std::max(2, static_cast<int>(std::llround(value)));
      } else if (parts[0] == "stability_tie_band") {
        loaded.stability_tie_band = std::max(0.0, value);
      } else if (parts[0] == "min_cluster_size") {
        loaded.min_cluster_size = static_cast<int>(std::llround(value));
      } else if (parts[0] == "global_threshold_ablation") {
        loaded.global_threshold_ablation = std::llround(value) != 0;
      } else if (parts[0] == "experimental_local_relabel") {
        loaded.experimental_local_relabel = std::llround(value) != 0;
      } else if (parts[0] == "experimental_incremental_geometry") {
        loaded.experimental_incremental_geometry = std::llround(value) != 0;
      } else if (parts[0] == "experimental_local_edge_refresh") {
        loaded.experimental_local_edge_refresh = std::llround(value) != 0;
      } else if (parts[0] == "experimental_local_bucket_refresh") {
        loaded.experimental_local_bucket_refresh = std::llround(value) != 0;
      } else if (parts[0] == "experimental_component_witness") {
        loaded.experimental_component_witness = std::llround(value) != 0;
      } else if (parts[0] == "experimental_connectivity_summary") {
        loaded.experimental_connectivity_summary = std::llround(value) != 0;
      } else if (parts[0] == "experimental_small_side_split") {
	        loaded.experimental_small_side_split = std::llround(value) != 0;
	      }
	    }
	    if (normalized_threshold_mode(loaded) == "bucket_percentile")
	      loaded.explicit_threshold = -1.0;
	    options_ = loaded;
	  }
  fit_csv(dir + "/points.csv");
}

} // namespace delaucluster
