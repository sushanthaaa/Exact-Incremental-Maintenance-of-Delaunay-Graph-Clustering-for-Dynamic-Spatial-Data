#ifndef DELAUNAY_CLUSTERER_H
#define DELAUNAY_CLUSTERER_H

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace delaucluster {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using VertexBase =
    CGAL::Triangulation_vertex_base_with_info_2<std::size_t, Kernel>;
using TriangulationDataStructure =
    CGAL::Triangulation_data_structure_2<VertexBase>;
using Delaunay =
    CGAL::Delaunay_triangulation_2<Kernel, TriangulationDataStructure>;
using Point = Delaunay::Point;
using VertexHandle = Delaunay::Vertex_handle;

struct PointRecord {
  std::size_t id = 0;
  double x = 0.0;
  double y = 0.0;
  int label = -1;
  bool has_label = false;
  int cluster_id = -1;
  bool is_noise = false;
  bool active = true;
};

struct EdgeRecord {
  std::size_t id = 0;
  std::size_t u = 0;
  std::size_t v = 0;
  double length = 0.0;
  double normalized_score = 0.0;
  double threshold_cap = std::numeric_limits<double>::quiet_NaN();
  double effective_threshold = std::numeric_limits<double>::quiet_NaN();
  bool retained = false;
};

struct ClusterOptions {
  int min_cluster_size = 0;
  double threshold_iqr_multiplier = 0.0;
  double explicit_threshold = -1.0;
  std::string threshold_mode = "local_scale";
  double bucket_threshold_percentile = 0.75;
  int bucket_min_scores = 0;
  int bucket_max_radius = 0;
  int knn_scale_k = 0;
  double supported_merge_factor = 1.20;
  int supported_merge_min_edges = 8;
  double saddle_cut_relax_factor = 1.05;
  double saddle_cut_cross_factor = 0.40;
  int saddle_cut_min_clusters = 12;
  double saddle_cut_max_cross_retained = 0.05;
  double manifold_core_factor = 0.85;
  double manifold_bridge_factor = 2.20;
  double manifold_alignment_min = 0.85;
  double manifold_anisotropy_min = 0.35;
  double manifold_dataset_anisotropy_min = 0.42;
  int manifold_min_clusters = 6;
  double stability_tie_band = 1e-9;
  bool global_threshold_ablation = false;
  bool experimental_local_relabel = false;
  bool experimental_incremental_geometry = false;
  bool experimental_local_edge_refresh = false;
  bool experimental_local_bucket_refresh = false;
  bool experimental_component_witness = true;
  bool experimental_connectivity_summary = true;
  bool experimental_small_side_split = true;
  bool disable_labeling_guard = false;
  bool always_full_relabel = false;
};

struct ClusterStats {
  std::size_t point_count = 0;
  std::size_t edge_count = 0;
  std::size_t retained_edge_count = 0;
  std::size_t num_clusters = 0;
  std::size_t noise_count = 0;
  std::size_t bucket_rows = 0;
  std::size_t bucket_cols = 0;
  std::size_t bucket_nonempty_point_cells = 0;
  std::size_t bucket_nonempty_edge_cells = 0;
  std::size_t bucket_max_points = 0;
  std::size_t bucket_max_edges = 0;
  double bucket_mean_points_nonempty = 0.0;
  double bucket_mean_edges_nonempty = 0.0;
  double threshold = std::numeric_limits<double>::quiet_NaN();
};

struct ThresholdTraceRecord {
  std::size_t candidate_index = 0;
  double threshold = 0.0;
  bool selected = false;
  bool nontrivial = false;
  std::size_t active_points = 0;
  std::size_t edge_count = 0;
  std::size_t retained_edges = 0;
  double retained_edge_pct = 0.0;
  std::size_t num_clusters = 0;
  std::size_t noise_count = 0;
  double noise_pct = 0.0;
  std::size_t largest_cluster = 0;
  double largest_cluster_pct = 0.0;
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

struct DynamicUpdateReport {
  std::string operation;
  long long time_ns = 0;
  std::size_t affected_cells = 0;
  std::size_t affected_bucket_points = 0;
  std::size_t affected_bucket_edges = 0;
  std::size_t affected_vertices = 0;
  std::size_t affected_edges = 0;
  std::size_t changed_labels = 0;
  bool incremental_geometry_used = false;
  bool local_edge_refresh_used = false;
  bool local_edge_refresh_fallback = false;
  std::size_t edge_refresh_scope_vertices = 0;
  std::size_t edge_refresh_candidate_edges = 0;
  std::size_t edge_refresh_mismatches = 0;
  bool bucket_refresh_global = true;
  bool local_bucket_refresh_used = false;
  std::size_t bucket_refresh_cells = 0;
  std::size_t bucket_refresh_points = 0;
  std::size_t bucket_refresh_edges = 0;
  std::size_t bucket_refresh_mismatches = 0;
  bool local_relabel_used = false;
  std::size_t relabel_scope_vertices = 0;
  std::size_t relabel_search_vertices = 0;
  std::size_t retained_edges_removed = 0;
  std::size_t retained_edges_added = 0;
  bool small_side_split_used = false;
  bool point_delete_split_used = false;
  bool local_relabel_fallback = false;
  bool verification_checked = false;
  std::size_t verification_mismatches = 0;
};

struct BucketCell {
  std::vector<std::size_t> point_ids;
  std::vector<std::size_t> edge_ids;
};

class SpatialBucketIndex {
public:
  void clear();
  void build(const std::vector<PointRecord> &points,
             const std::vector<EdgeRecord> &edges);
  void rebuild_on_current_layout(const std::vector<PointRecord> &points,
                                 const std::vector<EdgeRecord> &edges);

  bool contains_point(double x, double y) const;
  int bucket_index(double x, double y) const;
  std::vector<int> segment_bucket_indices(double x0, double y0, double x1,
                                          double y1) const;
  void remove_point(std::size_t point_id);
  void upsert_point(std::size_t point_id, double x, double y);
  void remove_edge(std::size_t edge_id);
  void upsert_edge(std::size_t edge_id, double x0, double y0, double x1,
                   double y1);
  std::vector<int> cell_neighborhood(int cell_index, int radius = 1) const;
  int cell_row(int cell_index) const;
  int cell_col(int cell_index) const;
  const std::vector<BucketCell> &cells() const { return cells_; }
  const std::unordered_map<std::size_t, int> &point_to_cell() const {
    return point_to_cell_;
  }
  const std::unordered_map<std::size_t, std::vector<int>> &edge_to_cells()
      const {
    return edge_to_cells_;
  }
  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::size_t nonempty_point_cells() const;
  std::size_t nonempty_edge_cells() const;
  std::size_t max_points_per_cell() const;
  std::size_t max_edges_per_cell() const;
  double mean_points_per_nonempty_cell() const;
  double mean_edges_per_nonempty_cell() const;

private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  double min_x_ = 0.0;
  double max_x_ = 0.0;
  double min_y_ = 0.0;
  double max_y_ = 0.0;
  double step_x_ = 1.0;
  double step_y_ = 1.0;
  std::vector<BucketCell> cells_;
  std::unordered_map<std::size_t, int> point_to_cell_;
  std::unordered_map<std::size_t, std::vector<int>> edge_to_cells_;
};

class DelaunayClusterer {
public:
  DelaunayClusterer() = default;
  explicit DelaunayClusterer(ClusterOptions options) : options_(options) {}

  void set_options(const ClusterOptions &options) { options_ = options; }
  const ClusterOptions &options() const { return options_; }

  void clear();
  void fit(const std::vector<PointRecord> &points);
  void fit_csv(const std::string &csv_path);

  DynamicUpdateReport insert_point(double x, double y, int label = -1,
                                   bool has_label = false);
  DynamicUpdateReport remove_nearest(double x, double y);
  DynamicUpdateReport move_nearest(double old_x, double old_y, double new_x,
                                   double new_y);

  int assign_cluster(double x, double y) const;
  bool assign_is_noise(double x, double y) const;

  void write_clusters_csv(const std::string &path) const;
  void write_edges_csv(const std::string &path) const;
  void write_bucket_points_csv(const std::string &path) const;
  void write_bucket_edges_csv(const std::string &path) const;
  void write_threshold_trace_csv(const std::string &path) const;
  void write_metadata(const std::string &path) const;
  void save_model(const std::string &dir) const;
  void load_model(const std::string &dir);
  std::size_t verify_against_full_recompute() const;
  std::size_t verify_retained_edges_against_full_recompute() const;
  std::size_t verify_bucket_index_against_frozen_rebuild() const;

  const std::vector<PointRecord> &records() const { return records_; }
  const std::vector<EdgeRecord> &edges() const { return edges_; }
  const std::vector<ThresholdTraceRecord> &threshold_trace() const {
    return threshold_trace_;
  }
  const ClusterStats &stats() const { return stats_; }
  const SpatialBucketIndex &bucket_index() const { return bucket_index_; }

  static std::vector<PointRecord> load_csv(const std::string &csv_path);

private:
  void rebuild_model(std::size_t *changed_labels = nullptr);
  void refresh_model_from_current_triangulation(
      std::size_t *changed_labels = nullptr, bool skip_bucket_refresh = false);
  void refresh_bucket_index_and_stats();
  void refresh_bucket_stats_only();
  bool refresh_bucket_index_locally(
      const std::set<std::size_t> &touched_point_ids,
      DynamicUpdateReport &report);
  void refresh_bucket_index_after_dynamic(
      const std::set<std::size_t> &touched_point_ids,
      DynamicUpdateReport &report);
  void rebuild_model_with_local_relabel(
      const std::set<std::size_t> &seed_ids,
      const std::set<std::size_t> &active_added_ids,
      const std::set<std::size_t> &active_removed_ids,
      const std::unordered_map<std::size_t, std::vector<std::size_t>>
          &removed_old_retained_neighbors,
      std::size_t *changed_labels, std::size_t *relabel_scope_vertices,
      std::size_t *relabel_search_vertices,
      std::size_t *edge_refresh_scope_vertices = nullptr,
      std::size_t *edge_refresh_candidate_edges = nullptr,
      bool *edge_refresh_fallback = nullptr,
      bool triangulation_is_current = false, bool skip_bucket_refresh = false);
  void rebuild_triangulation();
  void extract_edges_and_scales();
  void choose_threshold_and_prune();
  void apply_supported_component_merge();
  void apply_saddle_cut();
  void apply_manifold_filter();
  void rebuild_edge_lookup();
  void register_edge_record(std::size_t edge_index);
  bool erase_edge_by_id(std::size_t edge_id);
  double select_automatic_threshold(const std::vector<double> &scores);
  void record_explicit_threshold_trace();
  void label_connected_components(std::size_t *changed_labels);
  bool retained_labeling_has_split() const;
  bool repair_labeling_if_inconsistent(std::size_t *changed_labels);
  void refresh_cluster_stats_from_labels(
      bool refresh_connectivity_summary = true);
  void refresh_retained_connectivity_summary();
  void refresh_retained_connectivity_summary_for_clusters(
      const std::set<int> &cluster_ids);
  int default_min_cluster_size() const;
  std::vector<std::size_t> active_ids() const;
  std::size_t nearest_active_id(double x, double y) const;
  std::vector<int> affected_cells_for_positions(
      const std::vector<std::pair<double, double>> &positions) const;
  void populate_bucket_impact(
      DynamicUpdateReport &report,
      const std::vector<std::pair<double, double>> &positions) const;
  std::set<std::pair<std::size_t, std::size_t>> retained_edge_keys() const;
  bool refresh_edges_scales_and_pruning_locally(
      const std::set<std::size_t> &seed_ids,
      std::size_t *scope_vertices, std::size_t *candidate_edges);
  std::size_t estimate_affected_vertices(double x, double y) const;
  std::size_t estimate_affected_edges(double x, double y) const;

  ClusterOptions options_;
  Delaunay dt_;
  std::vector<PointRecord> records_;
  std::unordered_map<std::size_t, std::size_t> id_to_index_;
  std::unordered_map<std::size_t, VertexHandle> vertex_handles_;
  std::vector<EdgeRecord> edges_;
  std::unordered_map<std::size_t, std::size_t> edge_index_by_id_;
  std::unordered_map<std::size_t, std::vector<std::size_t>>
      edge_ids_by_vertex_;
  std::vector<ThresholdTraceRecord> threshold_trace_;
  bool last_local_edge_refresh_tracked_ = false;
  std::set<std::size_t> last_bucket_removed_edge_ids_;
  std::set<std::size_t> last_bucket_upsert_edge_ids_;
  std::set<std::pair<std::size_t, std::size_t>>
      last_removed_retained_edges_;
  std::set<std::pair<std::size_t, std::size_t>>
      last_added_retained_edges_;
  std::unordered_map<std::size_t, double> local_scale_;
  bool saddle_cut_activated_ = false;
  std::size_t saddle_cut_base_clusters_ = 0;
  double saddle_cut_cross_retained_fraction_ =
      std::numeric_limits<double>::quiet_NaN();
  bool manifold_filter_activated_ = false;
  std::size_t manifold_filter_base_clusters_ = 0;
  double manifold_filter_mean_anisotropy_ =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t manifold_filter_retained_edge_count_ = 0;
  SpatialBucketIndex bucket_index_;
  ClusterStats stats_;
  std::map<int, std::size_t> cluster_sizes_;
  std::map<int, std::set<std::size_t>> cluster_members_;
  std::set<std::pair<std::size_t, std::size_t>> retained_bridge_edges_;
  std::set<std::size_t> retained_articulation_vertices_;
  std::map<int, std::set<std::pair<std::size_t, std::size_t>>>
      retained_bridge_edges_by_cluster_;
  std::map<int, std::set<std::size_t>>
      retained_articulation_vertices_by_cluster_;
  std::set<int> retained_connectivity_summary_clusters_;
  std::size_t last_relabel_removed_retained_edges_count_ = 0;
  std::size_t last_relabel_added_retained_edges_count_ = 0;
  bool last_small_side_split_used_ = false;
  bool last_point_delete_split_used_ = false;
  std::size_t next_id_ = 0;
  std::size_t next_edge_id_ = 0;
};

} // namespace delaucluster

#endif // DELAUNAY_CLUSTERER_H
