#ifndef MGL_LOUVAIN_CUH
#define MGL_LOUVAIN_CUH
#include "../graph/host_graph.h"
#include "../graph/gpu_graph.cuh"
#include "../common.h"
#include "../partition/edge_partition.cuh"
#include "../bin/BIN.cuh"
#include "../coarsen_graph/coarsen_graph_mg.cuh"

namespace louvain {
    // tau: teleportation probability for the map equation random walk (Infomap default 0.15).
    // tau=0 recovers the undirected closed-form flow and is the current implementation.
    void run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter, const int max_phases, const double tau = 0.0);
};


#endif //MGL_LOUVAIN_CUH
