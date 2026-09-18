#include "DelaunayClusterer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using delaucluster::ClusterOptions;
using delaucluster::DelaunayClusterer;
using delaucluster::PointRecord;

namespace {

void usage() {
  std::cout << "Usage:\n"
            << "  ./cluster static <input_csv> <output_dir> [options]\n"
            << "  ./cluster dynamic <base_csv> <stream_csv> <output_dir> [options]\n"
            << "  ./cluster assign <model_dir> <query_csv> <output_csv>\n\n"
            << "Options:\n"
            << "  --min-cluster-size N\n"
            << "  --threshold-multiplier X  widen robust auto-threshold candidates\n"
            << "  --threshold X\n"
            << "  --threshold-mode MODE  raw_length|local_scale|bucket_percentile|knn_median|supported_merge|saddle_cut|manifold_filter\n"
            << "  --bucket-percentile P  percentile for bucket_percentile mode\n"
            << "  --bucket-min-scores N  minimum local edge scores for bucket mode\n"
            << "  --bucket-max-radius R  maximum cell radius for bucket mode\n"
            << "  --knn-scale-k K        graph-neighborhood size for knn_median mode\n"
            << "  --supported-merge-factor X near-threshold edge factor for supported_merge\n"
            << "  --supported-merge-min-edges N minimum inter-component support edges\n"
            << "  --saddle-cut-relax-factor X same-side threshold factor for saddle_cut\n"
            << "  --saddle-cut-cross-factor X cross-saddle threshold factor\n"
            << "  --saddle-cut-min-clusters N activation fragmentation floor\n"
            << "  --saddle-cut-max-cross-retained X activation cross-edge ceiling\n"
            << "  --manifold-core-factor X core threshold factor for manifold_filter\n"
            << "  --manifold-bridge-factor X tangent-continuity edge factor\n"
            << "  --manifold-alignment-min X minimum edge/tangent alignment\n"
            << "  --manifold-anisotropy-min X minimum local PCA anisotropy\n"
            << "  --manifold-dataset-anisotropy-min X activation anisotropy floor\n"
            << "  --manifold-min-clusters N activation fragmentation floor\n"
            << "  --stability-tie-band X threshold-objective band for stability tie-breaks\n"
            << "  --global-threshold\n"
            << "  --incremental-geometry dynamic mode: use CGAL insert/remove handles\n"
            << "  --local-edge-refresh dynamic mode: experimental affected-edge scoring\n"
            << "  --local-bucket-refresh dynamic mode: mutate SRR bucket registry locally\n"
            << "  --local-relabel       dynamic mode: experimental local component relabel\n"
            << "  --no-component-witness dynamic ablation: skip preserved-partition witness\n"
            << "  --no-connectivity-summary dynamic ablation: skip bridge/articulation cache\n"
            << "  --no-small-side-split dynamic ablation: skip bounded split relabel shortcut\n"
            << "  --no-guard             dynamic ablation: skip the O(V+E) labeling guard (heuristic-only, NOT exact)\n"
            << "  --always-full-relabel  dynamic baseline: skip the local relabel+guard; unconditional full relabel after refresh (exact)\n"
            << "  --verify-dynamic       dynamic mode: compare each step to full recompute\n";
}

ClusterOptions parse_options(int start, int argc, char **argv,
                             bool *verify_dynamic = nullptr) {
  ClusterOptions options;
  for (int i = start; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--min-cluster-size" && i + 1 < argc) {
      options.min_cluster_size = std::stoi(argv[++i]);
    } else if (arg == "--threshold-multiplier" && i + 1 < argc) {
      options.threshold_iqr_multiplier = std::stod(argv[++i]);
    } else if (arg == "--threshold" && i + 1 < argc) {
      options.explicit_threshold = std::stod(argv[++i]);
    } else if (arg == "--threshold-mode" && i + 1 < argc) {
      options.threshold_mode = argv[++i];
      options.global_threshold_ablation = false;
    } else if (arg == "--bucket-percentile" && i + 1 < argc) {
      options.bucket_threshold_percentile = std::stod(argv[++i]);
    } else if (arg == "--bucket-min-scores" && i + 1 < argc) {
      options.bucket_min_scores = std::max(0, std::stoi(argv[++i]));
    } else if (arg == "--bucket-max-radius" && i + 1 < argc) {
      options.bucket_max_radius = std::max(0, std::stoi(argv[++i]));
    } else if (arg == "--knn-scale-k" && i + 1 < argc) {
      options.knn_scale_k = std::stoi(argv[++i]);
    } else if (arg == "--supported-merge-factor" && i + 1 < argc) {
      options.supported_merge_factor = std::max(1.0, std::stod(argv[++i]));
    } else if (arg == "--supported-merge-min-edges" && i + 1 < argc) {
      options.supported_merge_min_edges = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--saddle-cut-relax-factor" && i + 1 < argc) {
      options.saddle_cut_relax_factor = std::max(1.0, std::stod(argv[++i]));
    } else if (arg == "--saddle-cut-cross-factor" && i + 1 < argc) {
      options.saddle_cut_cross_factor = std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--saddle-cut-min-clusters" && i + 1 < argc) {
      options.saddle_cut_min_clusters = std::max(2, std::stoi(argv[++i]));
    } else if (arg == "--saddle-cut-max-cross-retained" && i + 1 < argc) {
      options.saddle_cut_max_cross_retained =
          std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--manifold-core-factor" && i + 1 < argc) {
      options.manifold_core_factor = std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--manifold-bridge-factor" && i + 1 < argc) {
      options.manifold_bridge_factor =
          std::max(options.manifold_core_factor, std::stod(argv[++i]));
    } else if (arg == "--manifold-alignment-min" && i + 1 < argc) {
      options.manifold_alignment_min =
          std::min(1.0, std::max(0.0, std::stod(argv[++i])));
    } else if (arg == "--manifold-anisotropy-min" && i + 1 < argc) {
      options.manifold_anisotropy_min =
          std::min(1.0, std::max(0.0, std::stod(argv[++i])));
    } else if (arg == "--manifold-dataset-anisotropy-min" &&
               i + 1 < argc) {
      options.manifold_dataset_anisotropy_min =
          std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--manifold-min-clusters" && i + 1 < argc) {
      options.manifold_min_clusters = std::max(2, std::stoi(argv[++i]));
    } else if (arg == "--stability-tie-band" && i + 1 < argc) {
      options.stability_tie_band = std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--global-threshold") {
      options.global_threshold_ablation = true;
      options.threshold_mode = "raw_length";
    } else if (arg == "--incremental-geometry") {
      options.experimental_incremental_geometry = true;
    } else if (arg == "--local-edge-refresh") {
      options.experimental_local_edge_refresh = true;
      options.experimental_local_relabel = true;
      options.experimental_incremental_geometry = true;
    } else if (arg == "--local-bucket-refresh") {
      options.experimental_local_bucket_refresh = true;
      options.experimental_local_edge_refresh = true;
      options.experimental_local_relabel = true;
      options.experimental_incremental_geometry = true;
    } else if (arg == "--local-relabel") {
      options.experimental_local_relabel = true;
    } else if (arg == "--no-component-witness") {
      options.experimental_component_witness = false;
    } else if (arg == "--no-connectivity-summary") {
      options.experimental_connectivity_summary = false;
    } else if (arg == "--no-small-side-split") {
      options.experimental_small_side_split = false;
    } else if (arg == "--no-guard") {
      options.disable_labeling_guard = true;
    } else if (arg == "--always-full-relabel") {
      options.always_full_relabel = true;
    } else if (arg == "--verify-dynamic" && verify_dynamic != nullptr) {
      *verify_dynamic = true;
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return options;
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> parts;
  std::string item;
  for (char ch : line) {
    if (ch == ',') {
      parts.push_back(item);
      item.clear();
    } else {
      item.push_back(ch);
    }
  }
  parts.push_back(item);
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

void write_dynamic_header(const std::string &path) {
  std::ofstream out(path);
  out << "operation,time_ns,affected_cells,affected_bucket_points,"
         "affected_bucket_edges,affected_vertices,affected_edges,"
         "changed_labels,incremental_geometry_used,local_edge_refresh_used,"
         "local_edge_refresh_fallback,edge_refresh_scope_vertices,"
         "edge_refresh_candidate_edges,edge_refresh_mismatches,"
         "bucket_refresh_global,local_bucket_refresh_used,"
         "bucket_refresh_cells,bucket_refresh_points,bucket_refresh_edges,"
         "bucket_refresh_mismatches,"
         "local_relabel_used,relabel_scope_vertices,"
         "relabel_search_vertices,"
         "retained_edges_removed,retained_edges_added,"
         "small_side_split_used,"
         "point_delete_split_used,"
         "local_relabel_fallback,"
         "verification_checked,verification_mismatches\n";
}

void append_dynamic_report(const std::string &path,
                           const delaucluster::DynamicUpdateReport &report) {
  std::ofstream out(path, std::ios::app);
  out << report.operation << "," << report.time_ns << ","
      << report.affected_cells << "," << report.affected_bucket_points << ","
      << report.affected_bucket_edges << ","
      << report.affected_vertices << "," << report.affected_edges << ","
      << report.changed_labels << ","
      << (report.incremental_geometry_used ? 1 : 0) << ","
      << (report.local_edge_refresh_used ? 1 : 0) << ","
      << (report.local_edge_refresh_fallback ? 1 : 0) << ","
      << report.edge_refresh_scope_vertices << ","
      << report.edge_refresh_candidate_edges << ","
      << report.edge_refresh_mismatches << ","
      << (report.bucket_refresh_global ? 1 : 0) << ","
      << (report.local_bucket_refresh_used ? 1 : 0) << ","
      << report.bucket_refresh_cells << ","
      << report.bucket_refresh_points << ","
      << report.bucket_refresh_edges << ","
      << report.bucket_refresh_mismatches << ","
      << (report.local_relabel_used ? 1 : 0) << ","
      << report.relabel_scope_vertices << ","
      << report.relabel_search_vertices << ","
      << report.retained_edges_removed << ","
      << report.retained_edges_added << ","
      << (report.small_side_split_used ? 1 : 0) << ","
      << (report.point_delete_split_used ? 1 : 0) << ","
      << (report.local_relabel_fallback ? 1 : 0) << ","
      << (report.verification_checked ? 1 : 0)
      << "," << report.verification_mismatches << "\n";
}

void append_checked_report(const std::string &path,
                           DelaunayClusterer &clusterer,
                           delaucluster::DynamicUpdateReport report,
                           bool verify_dynamic) {
  if (verify_dynamic) {
    report.verification_checked = true;
    report.edge_refresh_mismatches =
        clusterer.verify_retained_edges_against_full_recompute();
    report.bucket_refresh_mismatches =
        clusterer.verify_bucket_index_against_frozen_rebuild();
    report.verification_mismatches = clusterer.verify_against_full_recompute();
  }
  append_dynamic_report(path, report);
  if (verify_dynamic &&
      (report.verification_mismatches != 0 ||
       report.edge_refresh_mismatches != 0 ||
       report.bucket_refresh_mismatches != 0)) {
    throw std::runtime_error("Dynamic verification failed after " +
                             report.operation + ": " +
                             std::to_string(report.verification_mismatches) +
                             " mismatched point labels, " +
                             std::to_string(report.edge_refresh_mismatches) +
                             " mismatched retained edges, " +
                             std::to_string(report.bucket_refresh_mismatches) +
                             " mismatched bucket registry entries");
  }
}

void run_dynamic_stream(DelaunayClusterer &clusterer,
                        const std::string &stream_csv,
                        const std::string &log_csv,
                        bool verify_dynamic) {
  write_dynamic_header(log_csv);
  std::ifstream in(stream_csv);
  if (!in.is_open())
    throw std::runtime_error("Cannot open stream CSV: " + stream_csv);

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto parts = split_csv_line(line);
    if (parts.empty())
      continue;

    double first_numeric = 0.0;
    const bool first_is_numeric = parse_double(parts[0], first_numeric);
    if (!first_is_numeric) {
      const std::string op = parts[0];
      if (op == "operation")
        continue;
      if (op == "insert" && parts.size() >= 3) {
        double x = std::stod(parts[1]);
        double y = std::stod(parts[2]);
        int label = parts.size() >= 4 ? std::stoi(parts[3]) : -1;
        append_checked_report(log_csv, clusterer,
                              clusterer.insert_point(x, y, label,
                                                     parts.size() >= 4),
                              verify_dynamic);
      } else if ((op == "delete" || op == "remove") && parts.size() >= 3) {
        append_checked_report(
            log_csv, clusterer,
            clusterer.remove_nearest(std::stod(parts[1]), std::stod(parts[2])),
            verify_dynamic);
      } else if (op == "move" && parts.size() >= 5) {
        append_checked_report(
            log_csv, clusterer,
            clusterer.move_nearest(std::stod(parts[1]), std::stod(parts[2]),
                                   std::stod(parts[3]), std::stod(parts[4])),
            verify_dynamic);
      } else {
        throw std::runtime_error("Malformed dynamic stream row: " + line);
      }
    } else if (parts.size() >= 2) {
      double x = first_numeric;
      double y = std::stod(parts[1]);
      int label = parts.size() >= 3 ? std::stoi(parts[2]) : -1;
      append_checked_report(log_csv, clusterer,
                            clusterer.insert_point(x, y, label,
                                                   parts.size() >= 3),
                            verify_dynamic);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 1;
  }

  try {
    std::string mode = argv[1];
    if (mode == "static" && argc >= 4) {
      ClusterOptions options = parse_options(4, argc, argv);
      DelaunayClusterer clusterer(options);
      clusterer.fit_csv(argv[2]);
      clusterer.save_model(argv[3]);
      std::cout << "DelauCluster static complete\n"
                << "  points: " << clusterer.stats().point_count << "\n"
                << "  clusters: " << clusterer.stats().num_clusters << "\n"
                << "  noise: " << clusterer.stats().noise_count << "\n"
                << "  threshold: " << clusterer.stats().threshold << "\n";
      return 0;
    }

    if (mode == "dynamic" && argc >= 5) {
      bool verify_dynamic = false;
      ClusterOptions options = parse_options(5, argc, argv, &verify_dynamic);
      std::filesystem::create_directories(argv[4]);
      DelaunayClusterer clusterer(options);
      clusterer.fit_csv(argv[2]);
      if ((options.experimental_local_relabel ||
           options.experimental_local_edge_refresh ||
           options.experimental_local_bucket_refresh) &&
          options.explicit_threshold <= 0.0) {
        options.explicit_threshold = clusterer.stats().threshold;
        clusterer.set_options(options);
      }
      run_dynamic_stream(clusterer, argv[3],
                         std::string(argv[4]) + "/dynamic_log.csv",
                         verify_dynamic);
      clusterer.save_model(argv[4]);
      std::cout << "DelauCluster dynamic complete\n"
                << "  points: " << clusterer.stats().point_count << "\n"
                << "  clusters: " << clusterer.stats().num_clusters << "\n"
                << "  log: " << std::string(argv[4]) + "/dynamic_log.csv"
                << "\n";
      return 0;
    }

    if (mode == "assign" && argc == 5) {
      DelaunayClusterer clusterer;
      clusterer.load_model(argv[2]);
      auto queries = DelaunayClusterer::load_csv(argv[3]);
      std::ofstream out(argv[4]);
      if (!out.is_open())
        throw std::runtime_error("Cannot write assignment output");
      out << "x,y,cluster_id,is_noise\n";
      for (const auto &q : queries) {
        out << q.x << "," << q.y << "," << clusterer.assign_cluster(q.x, q.y)
            << "," << (clusterer.assign_is_noise(q.x, q.y) ? 1 : 0) << "\n";
      }
      return 0;
    }

    usage();
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
