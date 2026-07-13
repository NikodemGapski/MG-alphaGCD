#ifndef MGL_LOUVAIN_CUH
#define MGL_LOUVAIN_CUH
#include <string>
#include "../graph/host_graph.h"
#include "../graph/gpu_graph.cuh"
#include "../common.h"
#include "../partition/edge_partition.cuh"
#include "../bin/BIN.cuh"
#include "../coarsen_graph/coarsen_graph_mg.cuh"

namespace louvain {
    // tau: teleportation probability for the map equation random walk (Infomap default 0.15).
    // tau=0 recovers the undirected closed-form flow and is the current implementation.
    // out_path: if non-empty, write the flat original-vertex -> community map (one
    // "<vertex> <community>" line per vertex) so the partition can be scored
    // independently of the GPU's own codelength (tools/score_partition.py).
    // move_mode: "warp"   = one warp per vertex, global per-vertex hash (default; clean path)
    //            "binned" = the original degree-binned tile/block kernels (kept for A/B)
    void run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter, const int max_phases, const double tau = 0.0,
             const std::string &out_path = std::string(), const std::string &move_mode = std::string("warp"));
};


#endif //MGL_LOUVAIN_CUH
