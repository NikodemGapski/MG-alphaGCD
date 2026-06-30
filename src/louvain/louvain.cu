#include "../../include/louvain/louvain.cuh"
#include "../../include/bin/BIN.cuh"

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace louvain {

__global__ void
init_community_id(
    vertex_t* __restrict__ shared_device_community_ids,
    vertex_t begin_vertex_id,
    vertex_t end_vertex_id,
    vertex_t local_vertices
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (vertex_t vertex_id = begin_vertex_id + grid.thread_rank(); vertex_id < end_vertex_id;
         vertex_id += grid.num_threads()) {
        shared_device_community_ids[vertex_id - begin_vertex_id] = vertex_id;
    }
}

__global__ void reduce_vertices_weights(
    vertex_t local_vertex,
    vertex_t* private_device_offset,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    __shared__ vertex_t range[(1024 / 32) * 2];
    vertex_t* t_ranges = &range[(block.thread_index().x / 32) * 2];
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */
    weight_t w = 0;
    edge_t e;
    vertex_t vertex_id;
    edge_t begin_edge;
    edge_t end_edge;

    // a vertex is assigned to a tile32
    for (vertex_id = tile32_id_grid; vertex_id < local_vertex; vertex_id += tile32_num_grid) {
        w = 0;
        if (tile32.thread_rank() < 2) {
            t_ranges[tile32.thread_rank()] = private_device_offset[vertex_id + tile32.thread_rank()];
        }

        tile32.sync();

        begin_edge = t_ranges[0];
        end_edge = t_ranges[1];

        for(e = begin_edge + tile32.thread_rank(); e < end_edge; e += 32){
            w += private_device_edge_weight[e];
        }

        w = cg::reduce(tile32, w, cg::plus<weight_t>());

        if(tile32.thread_rank() == 0) {
            private_device_vertex_weight[vertex_id] = w;
        }

        tile32.sync();
    }
}

__device__ __inline__ void locating_vertex(
    int &pe_dst,
    vertex_t &neighbor_id,
    vertex_t* private_device_part_vertex_offset,
    int n_pes)
{
#pragma unroll
    for (int i = 0; i < n_pes; i++) {
        if (neighbor_id < private_device_part_vertex_offset[i + 1]) {
            pe_dst = i;
            neighbor_id = neighbor_id - private_device_part_vertex_offset[i];
            break;
        }
    }

}

template <typename T>
__global__ void copy(
    T* src,
    T* dst,
    vertex_t len)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t id = grid.thread_rank(); id < len; id += grid.num_threads()) {
        dst[id] = src[id];
    }
}

// ---------------------------------------------------------------------------
// Objective seam: the local-move gain function.
//
// MG-alphaGCD originally maximised modularity. This build optimises the
// two-level *map equation* (Infomap) instead, for undirected graphs with no
// teleportation (tau = 0), where the random-walk flows have closed forms:
//     p_vis[v] = k_v / 2m,  q_vis[i] = vol(i) / 2m,  q_out[i] = cut(i) / 2m.
// `move_gain<O>` returns a value that is *maximised* over candidate moves; for
// the map equation that is the codelength decrease (-dL), so the surrounding
// Louvain driver (which maximises a score and only accepts positive moves) is
// reused unchanged. Switch ACTIVE_OBJECTIVE back to Modularity to recover the
// original algorithm.
// ---------------------------------------------------------------------------

enum class Objective { Modularity, MapEquation };

// Active objective for the optimized (version 2) path.
constexpr Objective ACTIVE_OBJECTIVE = Objective::MapEquation;

// p*log2(p), with the entropy convention F(0) = 0.
__device__ __forceinline__ weight_t plogp(weight_t x) {
    return x > 0.0 ? x * log2(x) : 0.0;
}

// e_vn   : weight of edges from v to candidate community n   (hash_table_value)
// eici   : weight of edges from v to its own community m
// ki     : weighted degree of v
// aci    : vol(m) - ki   (source community weight, already reduced by ki)
// acj    : vol(n)        (candidate community weight)
// qout_m : exit (cut) weight of m, raw units
// qout_n : exit (cut) weight of n, raw units
// q_total: global sum of exit weights, raw units (= Sum_i cut_i)
// mass   : m  (i.e. half of 2m -- the kernels pass mass after `mass /= 2`)
template <Objective O>
__device__ __forceinline__ weight_t move_gain(
    weight_t e_vn, weight_t eici, weight_t ki,
    weight_t aci, weight_t acj,
    weight_t qout_m, weight_t qout_n, weight_t q_total, weight_t mass);

template <>
__device__ __forceinline__ weight_t move_gain<Objective::Modularity>(
    weight_t e_vn, weight_t eici, weight_t ki,
    weight_t aci, weight_t acj,
    weight_t /*qout_m*/, weight_t /*qout_n*/, weight_t /*q_total*/, weight_t mass)
{
    weight_t wij = e_vn;
    wij -= (eici - ki * (aci - acj) / (2. * mass));
    wij /= mass;
    return wij;
}

template <>
__device__ __forceinline__ weight_t move_gain<Objective::MapEquation>(
    weight_t e_vn, weight_t eici, weight_t ki,
    weight_t aci, weight_t acj,
    weight_t qout_m, weight_t qout_n, weight_t q_total, weight_t mass)
{
    const weight_t two_m  = 2.0 * mass;                 // = 2m
    const weight_t p_v    = ki / two_m;                 // p_vis[v]
    const weight_t qvis_m = (aci + ki) / two_m;         // q_vis[m] (vol incl. v)
    const weight_t qvis_n = acj / two_m;                // q_vis[n]
    const weight_t qom    = qout_m / two_m;             // q_out[m]
    const weight_t qon    = qout_n / two_m;             // q_out[n]
    const weight_t Qp     = q_total / two_m;            // Q = Sum_i q_out[i]
    // change in module exit probabilities (undirected, tau = 0)
    const weight_t dqm    = (2.0 * eici - ki) / two_m;  // v leaves m: dcut = 2*e_vm - k_v
    const weight_t dqn    = (ki - 2.0 * e_vn) / two_m;  // v joins n:  dcut = k_v - 2*e_vn

    const weight_t dL =
          (plogp(Qp + dqm + dqn) - plogp(Qp))
        - 2.0 * ( (plogp(qom + dqm) - plogp(qom))
                + (plogp(qon + dqn) - plogp(qon)) )
        + (plogp((qom + dqm) + (qvis_m - p_v)) - plogp(qom + qvis_m))
        + (plogp((qon + dqn) + (qvis_n + p_v)) - plogp(qon + qvis_n));

    return -dL;  // maximise the codelength decrease
}

// Directed map equation move gain using the proportional-teleportation objective:
//   q_out[i] = τ·q_vis[i]·(1-q_vis[i]) + (1-τ)·q_walk_out[i]
// consistent with apply_teleportation_q_out.
//
// e_to_m_norm = Σ_{v→w∈m} w(v,w)/s_out[v]  (normalised transition prob v→own module)
// e_to_n_norm = Σ_{v→w∈n} w(v,w)/s_out[v]  (normalised transition prob v→candidate)
// in_m_flow   = Σ_{u→v, u∈m} p[u]·w(u,v)/s_out[u]  (walk in-flow into v from own module)
// in_n_flow   = Σ_{u→v, u∈n} p[u]·w(u,v)/s_out[u]  (walk in-flow into v from candidate)
// q_vis_{m,n}: probability units from PageRank (mass=1 in directed mode).
//
// EXACT walk-out deltas (verified vs brute-force ΔL to ~1e-15, see
// tools/directed_gain_check.py): when v leaves m, the edges OTHER nodes of m had
// INTO v stop being internal -> q_walk_out[m] gains in_m_flow; symmetrically the
// edges nodes of n have into v become internal -> q_walk_out[n] loses in_n_flow.
//   Δqw[m] = -p·(1-em) + in_m_flow,   Δqw[n] = +p·(1-en) - in_n_flow
// The previous code dropped both in_*_flow terms ("p_from_in ≈ 0"); that made the
// gain disagree with ΔL on the *sign* of the move ~34% of the time at τ=0.15,
// which is the directed-τ<0.5 "stuck at identity" failure. Pass in_*_flow=0 to
// recover the old approximation.
__device__ __forceinline__ weight_t move_gain_directed_approx(
    weight_t p_vis_v,     // = ki  (PageRank of v)
    weight_t q_vis_m,     // = aci + ki  (q_vis[m] including v)
    weight_t q_vis_n,     // = acj       (q_vis[n] before move)
    weight_t qout_m,      // current q_out[m] (after apply_teleportation_q_out)
    weight_t qout_n,      // current q_out[n]
    weight_t q_total,     // Q = Σ_i q_out[i]
    weight_t e_to_m_norm, // normalised transition prob from v to m
    weight_t e_to_n_norm, // normalised transition prob from v to n
    double tau,
    weight_t in_m_flow = 0.,   // walk in-flow into v from m  (0 = old approximation)
    weight_t in_n_flow = 0.    // walk in-flow into v from n
) {
    const double p  = (double)p_vis_v;
    const double qm = (double)q_vis_m;  // q_vis[m] including v
    const double qn = (double)q_vis_n;
    const double qom = (double)qout_m;
    const double qon = (double)qout_n;
    const double Qt  = (double)q_total;
    const double tv  = tau;
    const double em  = (double)e_to_m_norm;
    const double en  = (double)e_to_n_norm;
    const double im  = (double)in_m_flow;
    const double in_ = (double)in_n_flow;

    // Exact deltas for q_out[i]=τ·qv·(1-qv)+(1-τ)·qw when v moves from m to n.
    // Δqw[m] = -p·(1-em) + in_m_flow,  Δqw[n] = +p·(1-en) - in_n_flow.
    const double dqm = tv * p * (2.0*qm - 1.0 - p) - (1.0-tv) * (p * (1.0 - em) - im);
    const double dqn = tv * p * (1.0 - 2.0*qn - p) + (1.0-tv) * (p * (1.0 - en) - in_);

    // q_vis after the move
    const double qvis_m_new = qm - p;
    const double qvis_n_new = qn + p;

    const auto F = [](double x) -> double { return x > 0. ? x * log2(x) : 0.; };
    const double dL =
          (F(Qt + dqm + dqn) - F(Qt))
        - 2.0 * ((F(qom + dqm) - F(qom)) + (F(qon + dqn) - F(qon)))
        + (F((qom + dqm) + qvis_m_new) - F(qom + qm))
        + (F((qon + dqn) + qvis_n_new) - F(qon + qn));

    return (weight_t)(-dL);
}

// Map equation: fill `shared_device_community_q_out_delta` (indexed by *global*
// module id) with this PE's partial exit weight, i.e. for every local vertex the
// sum of its edge weights that cross into a different module. Mirrors the
// inter-community edge loop of compute_modularity but accumulates per module.
__global__ void __launch_bounds__(1024, 1)
compute_community_q_out_local_atomic(
    vertex_t local_vertices,
    vertex_t total_vertices,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_q_out_delta,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;

    // zero the per-global-module scratch
    for (vertex_t i = grid.thread_rank(); i < total_vertices; i += grid.num_threads()) {
        shared_device_community_q_out_delta[i] = 0.;
    }
    grid.sync();

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    vertex_t neighbor_id;
    int pe_dst;
    weight_t w_ext;

    // each vertex is handled by one tile32; accumulate its external edge weight
    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {
        src_community_id = shared_device_community_ids[vertex_id];
        w_ext = 0.;
        for (e = private_device_offset[vertex_id] + tile32.thread_rank(); e < private_device_offset[vertex_id + 1]; e += 32) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                dst_community_id = shared_device_community_ids[neighbor_id];
            } else {
                dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);
            }
            if (src_community_id != dst_community_id) {
                w_ext += private_device_edge_weight[e];   // self-loops have src==dst -> excluded
            }
        }
        w_ext = cg::reduce(tile32, w_ext, cg::plus<weight_t>());
        if (tile32.thread_rank() == 0 && w_ext != 0.) {
            atomicAdd(shared_device_community_q_out_delta + src_community_id, w_ext);
        }
    }
}

// Directed map equation: compute per-module exit WALK probability q_walk_out[i]
// = Σ_{v∈i} p_vis[v] * Σ_{v→w, w∉i} (w(v,w) / s_out[v]).
// Result is accumulated into q_out_delta (global module index).
// p_vis = private_device_vertex_weight (local, probability units).
// s_out = private_device_s_out (NVSHMEM, local index).
__global__ void __launch_bounds__(1024, 1)
compute_community_q_walk_out_local_atomic(
    vertex_t local_vertices,
    vertex_t total_vertices,
    vertex_t* private_device_offset,
    edge_t*   private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* p_vis,
    weight_t* s_out,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_q_out_delta,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;
    int tile32_id_grid = (block.num_threads() * block.group_index().x + block.thread_index().x) / 32;

    for (vertex_t i = grid.thread_rank(); i < total_vertices; i += grid.num_threads())
        shared_device_community_q_out_delta[i] = 0.;
    grid.sync();

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    vertex_t neighbor_id;
    int pe_dst;

    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {
        src_community_id = shared_device_community_ids[vertex_id];
        weight_t pv = p_vis[vertex_id];
        weight_t sv = s_out[vertex_id];
        weight_t walk_ext = 0.;

        if (sv > 0.) {
            for (e = private_device_offset[vertex_id] + tile32.thread_rank();
                 e < private_device_offset[vertex_id + 1]; e += 32) {
                neighbor_id = private_device_edge[e];
                locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
                dst_community_id = (pe_dst == my_pe)
                    ? shared_device_community_ids[neighbor_id]
                    : nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);
                if (src_community_id != dst_community_id) {
                    walk_ext += (weight_t)((double)private_device_edge_weight[e] / (double)sv * (double)pv);
                }
            }
        }
        walk_ext = cg::reduce(tile32, walk_ext, cg::plus<weight_t>());
        if (tile32.thread_rank() == 0 && walk_ext != 0.)
            atomicAdd(shared_device_community_q_out_delta + src_community_id, walk_ext);
    }
}

// Directed map equation: convert q_walk_out → q_out using proportional teleportation.
//   q_out[i] = τ·q_vis[i]·(1 - q_vis[i]) + (1-τ)·q_walk_out[i]
// P(exit i via teleport | visiting i) = τ·(1-q_vis[i]) (proportional model: teleport target
// is drawn proportional to ergodic visit probability, so fraction landing outside i = 1-q_vis[i]).
// This gives q_out→0 for the all-in-one community (q_vis=1, q_walk=0) and q_out≈p_v for
// singletons (q_vis=p_v≪1), guaranteeing L≥0 from compute_codelength.
__global__ void __launch_bounds__(1024, 1)
apply_teleportation_q_out(
    vertex_t local_vertices,
    weight_t tau,
    weight_t* shared_device_community_weight,   // q_vis[i]
    weight_t* shared_device_community_q_out      // in: q_walk_out[i], out: q_out[i]
)
{
    for (vertex_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < local_vertices; i += blockDim.x * gridDim.x) {
        weight_t q_vis  = shared_device_community_weight[i];
        weight_t q_walk = shared_device_community_q_out[i];
        shared_device_community_q_out[i] = (weight_t)(
            tau * (double)q_vis * (1.0 - (double)q_vis) + (1.0 - tau) * (double)q_walk);
    }
}

// Map equation: reduce the per-global-module partial exit weights across PEs into
// the owning PE's `shared_device_community_q_out` (local module range). Mirrors
// compute_community_weight, but assigns (zero + sum) instead of accumulating.
__global__ void __launch_bounds__(1024, 1)
compute_community_q_out(
    vertex_t local_vertices,
    vertex_t total_vertices,
    weight_t* shared_device_community_q_out,
    weight_t* shared_device_community_q_out_delta,
    vertex_t* private_device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    int i;
    int j;
    int nelems;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile_num_grid = grid.num_threads() / 32;
    int tile_id_grid = grid.thread_rank() / 32;
    int shared_offset = (block.thread_rank() / 32) * 32;
    vertex_t start_vertex_id = private_device_part_vertex_offset[my_pe];

    __shared__ double buff[1024];

    // zero the owning-PE local q_out range
    for (vertex_t v = grid.thread_rank(); v < local_vertices; v += grid.num_threads()) {
        shared_device_community_q_out[v] = 0.;
    }
    grid.sync();

    int local_tile;
    int remote_tile;
    if (n_pes == 1) {
        local_tile = tile_num_grid;
        remote_tile = 0;
    } else {
        local_tile = tile_num_grid / n_pes;
        remote_tile = tile_num_grid - local_tile;
    }

    // local workload: this PE's own partial contribution
    if (tile_id_grid < local_tile) {
        for (j = tile_id_grid * tile32.num_threads() + tile32.thread_rank(); j < local_vertices; j += (local_tile * tile32.num_threads())) {
            atomicAdd(shared_device_community_q_out + j, shared_device_community_q_out_delta[start_vertex_id + j]);
        }
    }
    // remote workload: other PEs' partial contributions
    else {
        for (vertex_t tile_offset = (tile_id_grid - local_tile) * tile32.num_threads(); tile_offset < local_vertices; tile_offset += (remote_tile * tile32.num_threads())) {
            nelems = min(tile32.num_threads(), local_vertices - tile_offset);
            for (i = 1; i < n_pes; i++) {
                nvshmemx_double_get_warp(buff + shared_offset, shared_device_community_q_out_delta + start_vertex_id + tile_offset, nelems, (my_pe + i) % n_pes);
                for (j = tile32.thread_rank(); j < nelems; j += tile32.num_threads()) {
                    atomicAdd(shared_device_community_q_out + tile_offset + j, buff[shared_offset + j]);
                }
            }
        }
    }
}

// Map equation: assemble the two-level codelength L from the per-module exit
// weights (q_out) and volumes (community_weight) and the per-node visit rates,
// then write the maximisable score (-L) into Q and the raw exit sum into Q_sum.
//     L = F(Q) - 2*Sum_i F(q_out_i) - Sum_v F(p_vis_v) + Sum_i F(q_out_i + q_vis_i)
// with q_out_i = cut_i/2m, q_vis_i = vol_i/2m, p_vis_v = k_v/2m, F(x)=x*log2(x).
// `mass` is the original 2m (not halved). `cl_reduce` is 4 doubles of symmetric scratch.
__global__ void __launch_bounds__(1024, 1)
compute_codelength(
    weight_t mass,
    vertex_t local_vertices,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* private_device_vertex_weight,
    weight_t* cl_reduce,
    int my_pe,
    int n_pes,
    weight_t* Q,
    weight_t* Q_sum
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    if (grid.thread_rank() == 0) {
        cl_reduce[0] = 0.; cl_reduce[1] = 0.; cl_reduce[2] = 0.; cl_reduce[3] = 0.;
    }
    grid.sync();

    weight_t s_qsum = 0.;  // Sum_i cut_i           (raw)
    weight_t s_qout = 0.;  // Sum_i F(q_out_i)
    weight_t s_mod  = 0.;  // Sum_i F(q_out_i + q_vis_i)
    weight_t s_node = 0.;  // Sum_v F(p_vis_v)

    // per-module contributions (each PE owns modules [0, local_vertices))
    for (vertex_t i = grid.thread_rank(); i < local_vertices; i += grid.num_threads()) {
        weight_t cut = shared_device_community_q_out[i];
        weight_t vol = shared_device_community_weight[i];
        s_qsum += cut;
        s_qout += plogp(cut / mass);
        s_mod  += plogp((cut + vol) / mass);
    }
    // per-node contributions (partition independent within a phase)
    for (vertex_t v = grid.thread_rank(); v < local_vertices; v += grid.num_threads()) {
        s_node += plogp(private_device_vertex_weight[v] / mass);
    }

    s_qsum = cg::reduce(tile32, s_qsum, cg::plus<weight_t>());
    s_qout = cg::reduce(tile32, s_qout, cg::plus<weight_t>());
    s_mod  = cg::reduce(tile32, s_mod,  cg::plus<weight_t>());
    s_node = cg::reduce(tile32, s_node, cg::plus<weight_t>());
    if (tile32.thread_rank() == 0) {
        atomicAdd(cl_reduce + 0, s_qsum);
        atomicAdd(cl_reduce + 1, s_qout);
        atomicAdd(cl_reduce + 2, s_mod);
        atomicAdd(cl_reduce + 3, s_node);
    }
    grid.sync();

    // all-reduce the 4 partials across PEs
    if (block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, cl_reduce, cl_reduce, 4);
    }
    grid.sync();

    if (grid.thread_rank() == 0) {
        weight_t q_raw = cl_reduce[0];
        weight_t Qp    = q_raw / mass;
        weight_t L     = plogp(Qp) - 2.0 * cl_reduce[1] - cl_reduce[3] + cl_reduce[2];
        // The map-equation codelength is an entropy and must be >= 0. A negative value
        // signals an invalid objective (e.g. a mis-derived directed q_out); surface it
        // loudly instead of letting the driver chase a meaningless minimum.
        if (L < -1.0e-6) {
            printf("[WARN] compute_codelength: negative L=%.9f (q_raw=%.9f mass=%.9f) -- invalid objective\n",
                   (double)L, (double)q_raw, (double)mass);
        }
        Q[0]     = -L;      // score: the driver maximises this (== minimising L)
        Q_sum[0] =  q_raw;  // raw Sum_i cut_i, read by the gain kernels
    }
    grid.sync();
}

// NOTE: compute_modularity below is the original objective and is no longer
// called on the version-2 path (superseded by compute_codelength); kept for
// reference / the Objective::Modularity template instantiation.
__global__ void __launch_bounds__(1024, 1)
compute_modularity(
    weight_t mass,
    vertex_t local_vertices,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_sum_community_weight,
    weight_t* shared_sum_internal_edge_weight,
    int my_pe,
    int n_pes,
    weight_t* Q
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */

    // Calculate the sum of community weighted degree

    __shared__ weight_t local_sum_degree_squared_tmp[1024 / 32];

    block.sync();

    weight_t sum_degree_squared = 0.0;
    weight_t degree_thread;

    if (grid.thread_rank() == 0)
    {
        shared_sum_community_weight[0] = 0.0;
        shared_sum_internal_edge_weight[0] = 0.0;
    }

    grid.sync();

    for (vertex_t i = grid.thread_rank(); i < local_vertices; i += grid.num_threads()) {
        degree_thread = shared_device_community_weight[i];
        sum_degree_squared += (degree_thread * degree_thread);
    }

    sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>());

    if (tile32.thread_rank() == 0) {
        local_sum_degree_squared_tmp[tile32.meta_group_rank()] = sum_degree_squared;
    }

    block.sync();

    if (tile32.meta_group_rank() == 0) {
        sum_degree_squared = tile32.thread_rank() < tile32.meta_group_size() ? local_sum_degree_squared_tmp[tile32.thread_rank()] : 0.0;
        sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>()); // sum over block
        if(tile32.thread_rank() == 0){
            atomicAdd(shared_sum_community_weight, sum_degree_squared);
        }
    }

    grid.sync();

    if(block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, shared_sum_community_weight, shared_sum_community_weight, 1);
    }

    grid.sync();


    // Calculate the sum of the weights of inter-community edges

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    int pe_dst;
    vertex_t neighbor_id;
    sum_degree_squared = 0.0; // reuse this register

    // a vertex is assigned to a tile32
    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {
        src_community_id = shared_device_community_ids[vertex_id];

        for (e = private_device_offset[vertex_id] + tile32.thread_rank(); e < private_device_offset[vertex_id + 1]; e += 32) {
            neighbor_id = private_device_edge[e];

            // which GPU this neighbor belongs to, neighbor_id is transformed to neighbor_id_local
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);

            if (pe_dst == my_pe) {
                dst_community_id = shared_device_community_ids[neighbor_id];
            } else {
                dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);
            }

            if (src_community_id == dst_community_id) {
                sum_degree_squared += private_device_edge_weight[e];
            }
        }
    }

    sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>());

    if (tile32.thread_rank() == 0) {
        local_sum_degree_squared_tmp[tile32.meta_group_rank()] = sum_degree_squared;
    }

    block.sync();

    if (tile32.meta_group_rank() == 0) {
        sum_degree_squared = tile32.thread_rank() < tile32.meta_group_size() ? local_sum_degree_squared_tmp[tile32.thread_rank()] : 0.0;
        sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>()); // sum over block
        if(tile32.thread_rank() == 0){
            atomicAdd(shared_sum_internal_edge_weight, sum_degree_squared);
        }
    }

    grid.sync();

    if(block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, shared_sum_internal_edge_weight, shared_sum_internal_edge_weight, 1);
    }

    grid.sync();

    // compute Q (modularity)
    if (grid.thread_rank() == 0) {
        Q[0] = shared_sum_internal_edge_weight[0] / mass - shared_sum_community_weight[0] / (mass * mass);
    }

    grid.sync();
}

__global__ void
calculate_eicj_and_move_vertex_hash_gl(
    vertex_t local_vertices,
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    vertex_t* hash_table_key,
    weight_t* hash_table_value,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t eici;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity;
    weight_t aci;
    weight_t acj;
    int pe_dst;
    vertex_t neighbor_id;

    vertex_t hash_table_lb;
    vertex_t hash_table_rb;
    vertex_t edge_lb;
    vertex_t edge_rb;
    long long unsigned int tmp;

    mass = mass / 2.0;

    __shared__ vertex_t neighbor_community_num[1024 / 32];

    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {

        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        weight_t ki =  private_device_vertex_weight[vertex_id];
        edge_lb = private_device_offset[vertex_id];
        edge_rb = private_device_offset[vertex_id + 1];
        hash_table_lb = edge_lb * 2;   // The space allocated to each vertex in the hash table is equal to twice the number of its adjacent edges.
        hash_table_rb = edge_rb * 2;

        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_rb; e += 32) {
            hash_table_key[e] = UINT32_MAX;
            hash_table_value[e] = 0.;
        }

        if (tile32.thread_rank() == 0) {
            neighbor_community_num[tile32.meta_group_rank()] = 0;

            dst_community_id = src_community_id; // In order to reuse the register dst_community_id
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
            }
        }

        aci = tile32.shfl(aci, 0);  // broadcast aci
        aci -= ki;

        for (e = edge_lb + tile32.thread_rank(); e < edge_rb; e += 32) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            tmp = dst_community_id * 107;
            hash = tmp % (hash_table_rb - hash_table_lb);

            if ((up_down && src_community_id > dst_community_id) || (!up_down && src_community_id < dst_community_id)) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij);
                        break;
                    }
                    hash = (hash + 1) % (hash_table_rb - hash_table_lb);
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij;
            }
        }

        tile32.sync();

        // reduce eici
        eici = cg::reduce(tile32, eici, cg::plus<weight_t>());

        // shuffle hash table
        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_rb; e += 32) {
            if (hash_table_key[e] != UINT32_MAX) {
                old_tmp = atomicAdd(neighbor_community_num + tile32.meta_group_rank(), 1);
                hash_table_key[hash_table_lb + old_tmp] = hash_table_key[e];
                hash_table_value[hash_table_lb + old_tmp] = hash_table_value[e];
            }
        }

//        tile32.sync();

        dst_community_id = src_community_id;

        // move vertex based on best modularity gain
        // iterate all neighbor community
        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_lb + neighbor_community_num[tile32.meta_group_rank()]; e += 32) {

            src_community_id = hash_table_key[e];   // neighbor community id
            old_tmp = src_community_id;
            locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                acj = shared_device_community_weight[old_tmp];
            } else {
                acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
            }

            wij = hash_table_value[e];
            wij -= (eici - ki * (aci - acj) / (2. * mass));
            wij /= mass;
            if (wij > best_modularity) {
                dst_community_id = src_community_id;
                best_modularity = wij;
            }
        }

        // Aggregate the dst_community_id and best_modularity results for tile32, and store the results in the thread with lane_id = 0
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            if (best_modularity_tmp > best_modularity) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        }

        if (tile32.thread_rank() == 0) {
            shared_device_community_ids_new_[vertex_id] = dst_community_id;
        }

        tile32.sync();
    }

}

template <int TILE_THREADS>
__global__ void calculate_eicj_and_move_vertex_sh_tile(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* Q_sum,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    weight_t* s_out_local,   // nullptr = undirected; local-index s_out array (NVSHMEM)
    double tau,              // teleportation probability (ignored when s_out_local==nullptr)
    vertex_t* in_offset,     // directed in-CSR (local-indexed); in_edge holds GLOBAL src ids
    edge_t* in_edge,
    weight_t* in_edge_weight
)
{
    int hash_len_tile = TILE_THREADS;   // the size of hash table that each tile has

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_THREADS>(block);
    int tile_num_grid = grid.num_threads() / TILE_THREADS;      /* the total number of tile in the whole grid */
    int tile_id_grid = grid.thread_rank() / TILE_THREADS;      /* the global id of the tile in the grid */

    vertex_t vertex_id;
    vertex_t tile_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t src_community_id_copy;
    vertex_t dst_community_id;
    weight_t eici;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity;
    weight_t aci;
    weight_t acj;
    weight_t qout_m;
    weight_t qout_n;
    int pe_dst;
    vertex_t neighbor_id;

    vertex_t hash_table_lb = (block.thread_rank() / TILE_THREADS) * hash_len_tile;
    vertex_t hash_table_rb = (block.thread_rank() / TILE_THREADS + 1) * hash_len_tile;
    vertex_t edge_lb;
    vertex_t edge_rb;
    long long unsigned int tmp;

    bool directed = (s_out_local != nullptr);
    if (!directed) mass = mass / 2.0;  // undirected uses halved mass; directed has mass=1
    weight_t q_total = Q_sum[0];

    // dynamic shared memory, |hash| = (512 / TILE_THREADS) * hash_len_tile = 512
    extern  __shared__ unsigned char shared_memory[];

    vertex_t* hash_table_key = (vertex_t*) shared_memory;
    weight_t* hash_table_value = (weight_t*) &hash_table_key[(block.num_threads() / TILE_THREADS) * hash_len_tile];
    // parallel slot array: walk in-flow into v from each candidate community (directed)
    weight_t* hash_table_in_value = (weight_t*) &hash_table_value[(block.num_threads() / TILE_THREADS) * hash_len_tile];


    for (tile_id = tile_id_grid; tile_id < bin_size; tile_id += tile_num_grid) {

        vertex_id = bin_permutation[bin_offset + tile_id];
        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        src_community_id_copy = src_community_id;
        weight_t ki =  private_device_vertex_weight[vertex_id];
        weight_t sv = directed ? s_out_local[vertex_id] : (weight_t)1.0;
        weight_t in_m = 0.;   // walk in-flow into v from its OWN module (directed)
        edge_lb = private_device_offset[vertex_id];
        edge_rb = private_device_offset[vertex_id + 1];

        for (e = hash_table_lb + tile.thread_rank(); e < hash_table_rb; e += TILE_THREADS) {
            hash_table_key[e] = UINT32_MAX;
            hash_table_value[e] = 0.;
            hash_table_in_value[e] = 0.;
        }

        tile.sync();

        if (tile.thread_rank() == 0) {

            dst_community_id = src_community_id; // In order to reuse the register dst_community_id
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
                qout_m = shared_device_community_q_out[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
                qout_m = nvshmem_double_g(shared_device_community_q_out + dst_community_id, pe_dst);
            }
        }

        tile.sync();

        aci = tile.shfl(aci, 0);  // broadcast aci
        aci -= ki;
        qout_m = tile.shfl(qout_m, 0);  // broadcast qout_m

        for (e = edge_lb + tile.thread_rank(); e < edge_rb; e += TILE_THREADS) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            tmp = dst_community_id * 107;
            hash = tmp % (hash_table_rb - hash_table_lb);

            weight_t wij_norm = (directed && sv > 0.) ? wij / sv : wij;
            if (src_community_id != dst_community_id) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij_norm);
                        break;
                    }
                    hash = (hash + 1) % (hash_table_rb - hash_table_lb);
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij_norm;
            }
        }

        tile.sync();

        // reduce eici
        eici = cg::reduce(tile, eici, cg::plus<weight_t>());

        // Directed: accumulate the walk in-flow into v from each module, so the gain
        // can use the EXACT q_walk_out deltas (see move_gain_directed_approx). For each
        // local in-neighbour u: contrib = p[u]*w(u,v)/s_out[u]; route by comm[u] into the
        // own-module scalar in_m, or (via hash LOOKUP, never insert -> no overflow) into
        // the candidate slot's in-value. Remote in-neighbours lack a symmetric p[u] and
        // are skipped (exact at -np 1, the directed regime that is actually used).
        if (directed) {
            for (e = in_offset[vertex_id] + tile.thread_rank(); e < in_offset[vertex_id + 1]; e += TILE_THREADS) {
                vertex_t u = in_edge[e];
                int pe_u; locating_vertex(pe_u, u, private_device_part_vertex_offset, n_pes);
                if (pe_u != my_pe) continue;
                weight_t su = s_out_local[u];
                if (su <= 0.) continue;
                weight_t contrib = (weight_t)((double)private_device_vertex_weight[u]
                                              * (double)in_edge_weight[e] / (double)su);
                vertex_t cu = shared_device_community_ids[u];
                if (cu == src_community_id_copy) {
                    in_m += contrib;
                } else {
                    vertex_t h = (vertex_t)(((long long unsigned)cu * 107) % (hash_table_rb - hash_table_lb));
                    while (hash_table_key[hash_table_lb + h] != UINT32_MAX) {
                        if (hash_table_key[hash_table_lb + h] == cu) {
                            atomicAdd(hash_table_in_value + hash_table_lb + h, contrib);
                            break;
                        }
                        h = (h + 1) % (hash_table_rb - hash_table_lb);
                    }
                }
            }
            tile.sync();
            in_m = cg::reduce(tile, in_m, cg::plus<weight_t>());
        }

        dst_community_id = src_community_id;

        // move vertex based on best modularity gain
        // iterate all neighbor community
        for (e = hash_table_lb + tile.thread_rank(); e < hash_table_rb; e += TILE_THREADS) {
            if (hash_table_key[e] != UINT32_MAX) {
                src_community_id = hash_table_key[e];   // neighbor community id
                old_tmp = src_community_id;
                locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
                if (pe_dst == my_pe) {
                    acj = shared_device_community_weight[old_tmp];
                    qout_n = shared_device_community_q_out[old_tmp];
                } else {
                    acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
                    qout_n = nvshmem_double_g(shared_device_community_q_out + old_tmp, pe_dst);
                }

                if (directed) {
                    wij = move_gain_directed_approx(ki, aci + ki, acj, qout_m, qout_n, q_total,
                                                    eici, hash_table_value[e], tau,
                                                    in_m, hash_table_in_value[e]);
                    // For directed mode do not break ties by community ID: on unweighted
                    // graphs every neighbour has identical gain, so ID tiebreaking causes
                    // all vertices to pile to vertex-0 (up_down=true) or vertex-N
                    // (up_down=false), creating one giant community that increases L.
                    // Letting the hash-table insertion order decide (first-found wins)
                    // creates a diverse, graph-structure-driven partition instead.
                    if (wij > best_modularity) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                } else {
                    wij = move_gain<ACTIVE_OBJECTIVE>(hash_table_value[e], eici, ki, aci, acj, qout_m, qout_n, q_total, mass);
                    if (up_down) {
                        if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                            dst_community_id = src_community_id;
                            best_modularity = wij;
                        }
                    } else {
                        if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                            dst_community_id = src_community_id;
                            best_modularity = wij;
                        }
                    }
                }
            }
        }

        tile.sync();

        // Aggregate the dst_community_id and best_modularity results for tile, and store the results in the thread with lane_id = 0
        for (e = TILE_THREADS / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile.shfl_down(dst_community_id, e);
            if (directed) {
                if (best_modularity_tmp > best_modularity) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }

        tile.sync();

        if (tile.thread_rank() == 0) {
            vertex_t final_dst;
            if (directed) {
                // No ID-based gate for directed mode: accept any positive-gain move.
                final_dst = dst_community_id;
            } else if (up_down) {
                final_dst = dst_community_id < src_community_id_copy ? dst_community_id : src_community_id_copy;
            } else {
                final_dst = dst_community_id > src_community_id_copy ? dst_community_id : src_community_id_copy;
            }
            shared_device_community_ids_new_[vertex_id] = final_dst;
        }

        tile.sync();
    }
}

template <int HASH_LEN>
__global__ void calculate_eicj_and_move_vertex_sh_bk(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* Q_sum,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t total_vertices,
    weight_t* s_out_local,
    double tau,
    vertex_t* in_offset,     // directed in-CSR (local-indexed); in_edge holds GLOBAL src ids
    edge_t* in_edge,
    weight_t* in_edge_weight
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num = tile32.meta_group_size();

    vertex_t vertex_id = block.group_index().x;
    vertex_t neighbor_id;

    edge_t e;
    vertex_t src_community_id;
    vertex_t src_community_id_copy;
    vertex_t dst_community_id;
    weight_t eici = 0.;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity = 0.0;
    weight_t aci;
    weight_t acj;
    weight_t qout_m;
    weight_t qout_n;
    int pe_dst;
    long long unsigned int tmp;
    bool directed = (s_out_local != nullptr);
    if (!directed) mass = mass / 2.0;
    weight_t q_total = Q_sum[0];

    __shared__ vertex_t hash_table_key[HASH_LEN];
    __shared__ weight_t hash_table_value[HASH_LEN];
    __shared__ weight_t hash_table_in_value[HASH_LEN];   // directed walk in-flow per slot
    __shared__ weight_t reduce_buffer[4];                // [3] = in_m (own-module in-flow)

    for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
        hash_table_key[e] = UINT32_MAX;
        hash_table_value[e] = 0.;
        hash_table_in_value[e] = 0.;
    }
    if (block.thread_rank() == 0) reduce_buffer[3] = 0.;

    if(vertex_id >= bin_size) return;

    vertex_id = bin_permutation[bin_offset + vertex_id];

    src_community_id = shared_device_community_ids[vertex_id];
    src_community_id_copy = src_community_id;
    weight_t ki =  private_device_vertex_weight[vertex_id];

    if (block.thread_rank() == 0) {
        dst_community_id = src_community_id;
        locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
        if (pe_dst == my_pe) {
            aci = shared_device_community_weight[dst_community_id];
            qout_m = shared_device_community_q_out[dst_community_id];
        } else {
            aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
            qout_m = nvshmem_double_g(shared_device_community_q_out + dst_community_id, pe_dst);
        }
        reduce_buffer[0] = aci;     // write result into shared memory which is visible to all threads within a block
        reduce_buffer[1] = 0.;
        reduce_buffer[2] = qout_m;
    }

    block.sync();

    // broadcast aci among block
    aci = reduce_buffer[0];
    aci -= ki;
    qout_m = reduce_buffer[2];
    block.sync();

    weight_t sv_bk = directed ? s_out_local[vertex_id] : (weight_t)1.0;
    for (e = private_device_offset[vertex_id] + block.thread_rank(); e < private_device_offset[vertex_id + 1]; e += block.num_threads()) {
        neighbor_id = private_device_edge[e];
        locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
        dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

        wij = private_device_edge_weight[e];
        weight_t wij_norm = (directed && sv_bk > 0.) ? wij / sv_bk : wij;
        tmp = dst_community_id * 107;
        hash = tmp % HASH_LEN;

        if (src_community_id != dst_community_id) {
            // hash table insert
            while (true) {
                old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, dst_community_id);
                if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                    atomicAdd(hash_table_value + hash, wij_norm);
                    break;
                }
                hash = (hash + 1) % HASH_LEN;
            }
        } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
            eici += wij_norm;
        }
    }

    block.sync();

    // reduce eici among block
    eici = cg::reduce(tile32, eici, cg::plus<weight_t>());      //reduce eici among tile32
    if (tile32.thread_rank() == 0) {
        atomicAdd(reduce_buffer + 1, eici);
    }
    block.sync();

    eici = reduce_buffer[1];

    // Directed: accumulate walk in-flow into v (own module -> reduce_buffer[3]; candidate
    // modules -> hash_table_in_value via lookup, never insert). Local in-neighbours only.
    if (directed) {
        weight_t in_m_local = 0.;
        for (e = in_offset[vertex_id] + block.thread_rank(); e < in_offset[vertex_id + 1]; e += block.num_threads()) {
            vertex_t u = in_edge[e];
            int pe_u; locating_vertex(pe_u, u, private_device_part_vertex_offset, n_pes);
            if (pe_u != my_pe) continue;
            weight_t su = s_out_local[u];
            if (su <= 0.) continue;
            weight_t contrib = (weight_t)((double)private_device_vertex_weight[u]
                                          * (double)in_edge_weight[e] / (double)su);
            vertex_t cu = shared_device_community_ids[u];
            if (cu == src_community_id_copy) {
                in_m_local += contrib;
            } else {
                vertex_t h = (vertex_t)(((long long unsigned)cu * 107) % HASH_LEN);
                while (hash_table_key[h] != UINT32_MAX) {
                    if (hash_table_key[h] == cu) { atomicAdd(hash_table_in_value + h, contrib); break; }
                    h = (h + 1) % HASH_LEN;
                }
            }
        }
        in_m_local = cg::reduce(tile32, in_m_local, cg::plus<weight_t>());
        if (tile32.thread_rank() == 0) atomicAdd(reduce_buffer + 3, in_m_local);
        block.sync();
    }
    weight_t in_m = reduce_buffer[3];

    dst_community_id = src_community_id;

    // move vertex based on best modularity gain
    // iterate all neighbor community
    for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
        if (hash_table_key[e] != UINT32_MAX) {
            src_community_id = hash_table_key[e];   // neighbor community id
            old_tmp = src_community_id;
            locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                acj = shared_device_community_weight[old_tmp];
                qout_n = shared_device_community_q_out[old_tmp];
            } else {
                acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
                qout_n = nvshmem_double_g(shared_device_community_q_out + old_tmp, pe_dst);
            }

            wij = directed
                ? move_gain_directed_approx(ki, aci + ki, acj, qout_m, qout_n, q_total, eici, hash_table_value[e], tau, in_m, hash_table_in_value[e])
                : move_gain<ACTIVE_OBJECTIVE>(hash_table_value[e], eici, ki, aci, acj, qout_m, qout_n, q_total, mass);
            // Add a constraint: when the best score is the same, choose the one with the smallest ID.
            if (up_down) {
                if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                    dst_community_id = src_community_id;
                    best_modularity = wij;
                }
            } else {
                if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                    dst_community_id = src_community_id;
                    best_modularity = wij;
                }
            }
        }
    }

    block.sync();

    //  Aggregate the dst_community_id and best_modularity results for block, and store the results in the thread with lane_id = 0

    // tile32-wide reduce result
    for (e = 32 / 2; e > 0; e /= 2) {
        weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
        vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
        // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
        if (up_down) {
            if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        } else {
            if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        }
    }

    tile32.sync();

    // write tile32-wide reduce result into shared memory
    if (tile32.thread_rank() == 0) {
        hash_table_key[tile32.meta_group_rank()] = dst_community_id;
        hash_table_value[tile32.meta_group_rank()] = best_modularity;
    }

    block.sync();

    if (block.thread_rank() < tile32_num) {
        best_modularity = hash_table_value[block.thread_rank()];
        dst_community_id = hash_table_key[block.thread_rank()];
    }

    block.sync();

    if (block.thread_rank() < 32) {
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
            if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }
    }

    block.sync();

    if (block.thread_rank() == 0) {
        if (up_down) {
            shared_device_community_ids_new_[vertex_id] = dst_community_id < src_community_id_copy ? dst_community_id : src_community_id_copy;
        } else {
            shared_device_community_ids_new_[vertex_id] = dst_community_id > src_community_id_copy ? dst_community_id : src_community_id_copy;
        }
    }
    block.sync();
}


__global__ void __launch_bounds__(1024, 1)
calculate_eicj_and_move_vertex_gl_bk(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* Q_sum,
    vertex_t* hash_table_key,
    weight_t* hash_table_value,
    weight_t* hash_table_in_value,   // global walk in-flow per slot (directed)
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    edge_t HASH_LEN,
    weight_t* s_out_local,
    double tau,
    vertex_t* in_offset,     // directed in-CSR (local-indexed); in_edge holds GLOBAL src ids
    edge_t* in_edge,
    weight_t* in_edge_weight
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num = tile32.meta_group_size();

    vertex_t bk_id = block.group_index().x;
    vertex_t vertex_id;
    vertex_t neighbor_id;

    edge_t e;
    vertex_t src_community_id;
    vertex_t src_community_id_copy;
    vertex_t dst_community_id;
    weight_t eici = 0.;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity = 0.0;
    weight_t aci;
    weight_t acj;
    weight_t qout_m;
    weight_t qout_n;
    int pe_dst;
    long long unsigned int tmp;
    bool directed = (s_out_local != nullptr);
    if (!directed) mass = mass / 2.0;
    weight_t q_total = Q_sum[0];

    __shared__ vertex_t buffer_int[1024/32];
    __shared__ weight_t buffer_double[1024/32];
    __shared__ weight_t in_m_buf;     // own-module walk in-flow (directed)

    vertex_t hash_table_lb = block.group_index().x * HASH_LEN;

    if(bk_id >= bin_size) return;

    for (; bk_id < bin_size; bk_id += grid.num_blocks()) {
        vertex_id = bin_permutation[bin_offset + bk_id];

        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        src_community_id_copy = src_community_id;
        weight_t ki =  private_device_vertex_weight[vertex_id];

        for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
            hash_table_key[hash_table_lb + e] = UINT32_MAX;
            hash_table_value[hash_table_lb + e] = 0.;
            hash_table_in_value[hash_table_lb + e] = 0.;
        }
        if (block.thread_rank() == 0) in_m_buf = 0.;
        block.sync();

        if (block.thread_rank() == 0) {
            dst_community_id = src_community_id;
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
                qout_m = shared_device_community_q_out[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
                qout_m = nvshmem_double_g(shared_device_community_q_out + dst_community_id, pe_dst);
            }
            buffer_double[0] = aci;     // write result into shared memory which is visible to all threads within a block
            buffer_double[1] = 0.;
            buffer_double[2] = qout_m;
        }
        block.sync();

        aci = buffer_double[0];
        aci -= ki;
        qout_m = buffer_double[2];
        block.sync();

        for (e = private_device_offset[vertex_id] + block.thread_rank(); e < private_device_offset[vertex_id + 1]; e += block.num_threads()) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            weight_t sv_gl = directed ? s_out_local[vertex_id] : (weight_t)1.0;
            weight_t wij_norm = (directed && sv_gl > 0.) ? wij / sv_gl : wij;
            tmp = dst_community_id * 107;
            hash = tmp % HASH_LEN;

            if (src_community_id != dst_community_id) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij_norm);
                        break;
                    }
                    hash = (hash + 1) % HASH_LEN;
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij_norm;
            }
        }

        block.sync();

        // reduce eici among block
        eici = cg::reduce(tile32, eici, cg::plus<weight_t>());      //reduce eici among tile32
        if (tile32.thread_rank() == 0) {
            atomicAdd(buffer_double + 1, eici);
        }
        block.sync();

        eici = buffer_double[1];

        // Directed: accumulate walk in-flow into v (own module -> in_m_buf; candidate
        // modules -> hash_table_in_value via lookup, never insert). Local in-neighbours only.
        if (directed) {
            weight_t in_m_local = 0.;
            for (e = in_offset[vertex_id] + block.thread_rank(); e < in_offset[vertex_id + 1]; e += block.num_threads()) {
                vertex_t u = in_edge[e];
                int pe_u; locating_vertex(pe_u, u, private_device_part_vertex_offset, n_pes);
                if (pe_u != my_pe) continue;
                weight_t su = s_out_local[u];
                if (su <= 0.) continue;
                weight_t contrib = (weight_t)((double)private_device_vertex_weight[u]
                                              * (double)in_edge_weight[e] / (double)su);
                vertex_t cu = shared_device_community_ids[u];
                if (cu == src_community_id_copy) {
                    in_m_local += contrib;
                } else {
                    vertex_t h = (vertex_t)(((long long unsigned)cu * 107) % HASH_LEN);
                    while (hash_table_key[hash_table_lb + h] != UINT32_MAX) {
                        if (hash_table_key[hash_table_lb + h] == cu) { atomicAdd(hash_table_in_value + hash_table_lb + h, contrib); break; }
                        h = (h + 1) % HASH_LEN;
                    }
                }
            }
            in_m_local = cg::reduce(tile32, in_m_local, cg::plus<weight_t>());
            if (tile32.thread_rank() == 0) atomicAdd(&in_m_buf, in_m_local);
            block.sync();
        }
        weight_t in_m = in_m_buf;

        dst_community_id = src_community_id;

//         move vertex based on best modularity gain
//         iterate all neighbor community
        for (e = hash_table_lb + block.thread_rank(); e < hash_table_lb + HASH_LEN; e += block.num_threads()) {
            if (hash_table_key[e] != UINT32_MAX) {
                src_community_id = hash_table_key[e];   // neighbor community id
                old_tmp = src_community_id;
                locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
                if (pe_dst == my_pe) {
                    acj = shared_device_community_weight[old_tmp];
                    qout_n = shared_device_community_q_out[old_tmp];
                } else {
                    acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
                    qout_n = nvshmem_double_g(shared_device_community_q_out + old_tmp, pe_dst);
                }

                wij = directed
                    ? move_gain_directed_approx(ki, aci + ki, acj, qout_m, qout_n, q_total, eici, hash_table_value[e], tau, in_m, hash_table_in_value[e])
                    : move_gain<ACTIVE_OBJECTIVE>(hash_table_value[e], eici, ki, aci, acj, qout_m, qout_n, q_total, mass);
                // Add a constraint: when the best score is the same, choose the one with the smallest ID.
                if (up_down) {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                } else {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                }
            }
        }

        block.sync();

        // tile32-wide reduce result
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
            if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }

        // write tile32-wide reduce result into shared memory
        if (tile32.thread_rank() == 0) {
            buffer_int[tile32.meta_group_rank()] = dst_community_id;
            buffer_double[tile32.meta_group_rank()] = best_modularity;
        }

        block.sync();

        if (block.thread_rank() < tile32_num) {
            best_modularity = buffer_double[block.thread_rank()];
            dst_community_id = buffer_int[block.thread_rank()];
        }

        block.sync();

        if (block.thread_rank() < 32) {
            for (e = 32 / 2; e > 0; e /= 2) {
                weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
                vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
                // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
                if (up_down) {
                    if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                        best_modularity = best_modularity_tmp;
                        dst_community_id = dst_community_id_tmp;
                    }
                } else {
                    if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                        best_modularity = best_modularity_tmp;
                        dst_community_id = dst_community_id_tmp;
                    }
                }
            }
        }

        block.sync();

        if (block.thread_rank() == 0) {
            if (up_down) {
                shared_device_community_ids_new_[vertex_id] = dst_community_id < src_community_id_copy ? dst_community_id : src_community_id_copy;
            } else {
                shared_device_community_ids_new_[vertex_id] = dst_community_id > src_community_id_copy ? dst_community_id : src_community_id_copy;
            }
        }

        block.sync();
    }
}

void calculate_eicj_and_move_vertex_bin(
    vertex_t local_vertices,
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* Q_sum,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    BIN *bins,
    cudaStream_t *streams,
    vertex_t total_vertices,
    cudaStream_t default_stream,
    weight_t* s_out_local,   // nullptr = undirected
    double tau,              // ignored when s_out_local == nullptr
    // directed in-CSR (local-indexed) for the exact walk-in-flow gain correction;
    // nullptr in undirected mode. in_edge[] holds GLOBAL source ids.
    vertex_t* in_offset = nullptr,
    edge_t* in_edge = nullptr,
    weight_t* in_edge_weight = nullptr
){
    int grid_num;
    int block_num;
    size_t d_shared_mem;

    vertex_t * hash_table_key = nullptr;
    weight_t * hash_table_value = nullptr;
    weight_t * hash_table_in_value = nullptr;   // global walk in-flow per slot (directed gl_bk)

    for (int i = BIN_NUM - 1; i >= 0; i--) {
        if (bins->bin_size[i] > 0) {
            switch (i) {
                case 0:
                    // tile2 for a vertex whose #edges is less than 2
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 2);
                    d_shared_mem = sizeof(vertex_t) * block_num + 2 * sizeof(weight_t) * block_num;  // +1 weight_t/slot for in-flow
                    calculate_eicj_and_move_vertex_sh_tile<2><<<grid_num, block_num, d_shared_mem, streams[0]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 1:
                    // tile4 for a vertex whose #edges is less than 4
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 4);
                    d_shared_mem = sizeof(vertex_t) * block_num + 2 * sizeof(weight_t) * block_num;  // +1 weight_t/slot for in-flow
                    calculate_eicj_and_move_vertex_sh_tile<4><<<grid_num, block_num, d_shared_mem, streams[1]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 2:
                    // tile8 for a vertex whose #edges is less than 8
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 8);
                    d_shared_mem = sizeof(vertex_t) * block_num + 2 * sizeof(weight_t) * block_num;  // +1 weight_t/slot for in-flow
                    calculate_eicj_and_move_vertex_sh_tile<8><<<grid_num, block_num, d_shared_mem, streams[2]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 3:
                    // tile16 for a vertex whose #edges is less than 16
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 16);
                    d_shared_mem = sizeof(vertex_t) * block_num + 2 * sizeof(weight_t) * block_num;  // +1 weight_t/slot for in-flow
                    calculate_eicj_and_move_vertex_sh_tile<16><<<grid_num, block_num, d_shared_mem, streams[3]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;

                case 4:
                    // tile32 for a vertex whose #edges is less than 32
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 32);
                    d_shared_mem = sizeof(vertex_t) * block_num + 2 * sizeof(weight_t) * block_num;  // +1 weight_t/slot for in-flow
                    calculate_eicj_and_move_vertex_sh_tile<32><<<grid_num, block_num, d_shared_mem, streams[4]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 5:
                    block_num = 128;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<128><<<grid_num, block_num, 0, streams[5]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            total_vertices, s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 6:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<512><<<grid_num, block_num, 0, streams[6]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            total_vertices, s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 7:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<2048><<<grid_num, block_num, 0, streams[7]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            total_vertices, s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 8:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<4094><<<grid_num, block_num, 0, streams[8]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            total_vertices, s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
                case 9:
                    int SMs = 80;
                    int hash_table_num = min(SMs, bins->bin_size[i]);
                    vertex_t max_degree = bins->max_degree * 2;
                    vertex_t len_hash_table = hash_table_num * max_degree;
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_key, sizeof(vertex_t) * len_hash_table));
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_value, sizeof(weight_t) * len_hash_table));
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_in_value, sizeof(weight_t) * len_hash_table));
                    grid_num = hash_table_num;
                    block_num = 1024;
                    calculate_eicj_and_move_vertex_gl_bk<<<grid_num, block_num, 0, streams[9]>>>(
                            begin_vertex_id,
                            private_device_offset, private_device_edge, private_device_edge_weight,
                            private_device_vertex_weight, private_device_part_vertex_offset,
                            shared_device_community_ids_new, shared_device_community_ids,
                            shared_device_community_weight, shared_device_community_q_out, Q_sum,
                            hash_table_key, hash_table_value, hash_table_in_value,
                            mass, my_pe, n_pes, up_down,
                            bins->bin_size[i], bins->bin_offset[i], bins->device_bin_permutation,
                            max_degree, s_out_local, tau, in_offset, in_edge, in_edge_weight);
                    break;
            }
        }
    }

    for (int i = 0; i < BIN_NUM; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    CUDA_RT_CALL(cudaFree(hash_table_key));
    CUDA_RT_CALL(cudaFree(hash_table_value));
    CUDA_RT_CALL(cudaFree(hash_table_in_value));
}


__global__ void __launch_bounds__(1024, 1)
compute_community_weight_local_atomic(
    vertex_t local_vertices,
    vertex_t total_vertices,
    weight_t* private_device_vertex_weight,
    vertex_t* shared_device_community_ids,
    vertex_t* shared_device_community_ids_new_,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_delta_weight,
    vertex_t* private_device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t vertex_weight;

    // init shared_device_community_delta_weight
    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        shared_device_community_delta_weight[vertex_id] = 0.;
    }

    grid.sync();

    // update local shared_device_community_delta_weight
    for (vertex_id = grid.thread_rank(); vertex_id < local_vertices; vertex_id += grid.num_threads()) {
        src_community_id = shared_device_community_ids[vertex_id];
        dst_community_id = shared_device_community_ids_new_[vertex_id];
        if (src_community_id != dst_community_id) {
            vertex_weight = private_device_vertex_weight[vertex_id];
            atomicAdd(shared_device_community_delta_weight + src_community_id, -vertex_weight);
            atomicAdd(shared_device_community_delta_weight + dst_community_id, vertex_weight);
        }
    }
}


__global__ void __launch_bounds__(1024, 1)
compute_community_weight(
    vertex_t local_vertices,
    vertex_t total_vertices,
    weight_t* private_device_vertex_weight,
    vertex_t* shared_device_community_ids,
    vertex_t* shared_device_community_ids_new_,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_delta_weight,
    vertex_t* private_device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    int i;
    int j;
    int nelems;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile_num_grid = grid.num_threads() / 32;      /* the total number of tile in the whole grid */
    int tile_id_grid = grid.thread_rank() / 32;      /* the global id of the tile in the grid */
    int shared_offset = (block.thread_rank() / 32) * 32;
    vertex_t start_vertex_id = private_device_part_vertex_offset[my_pe];

    __shared__ double buff[1024];

    int local_tile;
    int remote_tile;
    if (n_pes == 1) {
        local_tile = tile_num_grid;
        remote_tile = 0;
    } else {
        local_tile = tile_num_grid / n_pes;
        remote_tile = tile_num_grid - local_tile;
    }

    // local workload
    if (tile_id_grid < local_tile) {
        for (j = tile_id_grid * tile32.num_threads() + tile32.thread_rank(); j < local_vertices; j += (local_tile * tile32.num_threads())) {
            atomicAdd(shared_device_community_weight + j, shared_device_community_delta_weight[start_vertex_id + j]);
        }
    }

    // remote workload
    else {
        for (vertex_t tile_offset = (tile_id_grid - local_tile) * tile32.num_threads(); tile_offset < local_vertices; tile_offset += (remote_tile * tile32.num_threads())) {
            nelems = min(tile32.num_threads(), local_vertices - tile_offset);
            for (i = 1; i < n_pes; i++) {
                nvshmemx_double_get_warp(buff + shared_offset, shared_device_community_delta_weight + start_vertex_id + tile_offset, nelems, (my_pe + i) % n_pes);
                for (j = tile32.thread_rank(); j < nelems; j += tile32.num_threads()) {
                    atomicAdd(shared_device_community_weight + tile_offset + j, buff[shared_offset + j]);
                }
            }
        }
    }
}

} // namespace louvain

// File-scope device helper: determine which PE owns global vertex vid and
// convert vid to a local index.  Mirrors louvain::locating_vertex exactly.
__device__ __inline__ void locate_vertex_pe(
    int &pe, vertex_t &vid, vertex_t* part_vertex_offset, int n_pes)
{
    for (int i = 0; i < n_pes; i++) {
        if (vid < part_vertex_offset[i + 1]) {
            pe  = i;
            vid = vid - part_vertex_offset[i];
            break;
        }
    }
}

// One iteration of the distributed PageRank power iteration.
// p_cur[lv]  = current p_vis for local vertex lv (local index, indexed 0..local_vertices-1)
// p_new[lv]  = output p_vis after this step
// in_offset[lv], in_edge[e], in_edge_weight[e] = in-CSR for local vertices
//   (in_edge[e] is a GLOBAL source vertex id)
// s_out[lu]  = out-strength of local vertex lu on THIS PE (NVSHMEM symmetric)
// pr_reduce  = 2 doubles of NVSHMEM symmetric scratch: [dangling_mass, l1_norm]
// N          = total_vertices (normalizer for uniform teleportation)
__global__ void __launch_bounds__(1024, 1)
pagerank_step(
    vertex_t local_vertices,
    double   tau,
    double   inv_N,              // 1.0 / total_vertices
    weight_t *p_cur,             // NVSHMEM: remote PEs read p_cur[local_idx] from us
    weight_t *p_new,             // plain cudaMalloc: local output
    vertex_t *in_offset,
    edge_t   *in_edge,
    weight_t *in_edge_weight,
    weight_t *s_out,             // NVSHMEM: remote PEs read s_out[local_idx] from us
    vertex_t *part_vertex_offset,
    int       my_pe,
    int       n_pes,
    weight_t *pr_reduce          // NVSHMEM: [0]=dangling_mass, [1]=l1_norm
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    if (grid.thread_rank() == 0) {
        pr_reduce[0] = 0.;  // dangling mass accumulator
        pr_reduce[1] = 0.;  // l1 norm accumulator
    }
    grid.sync();

    // Phase 1: compute local dangling mass (vertices with s_out == 0)
    weight_t local_dangling = 0.;
    for (vertex_t lv = grid.thread_rank(); lv < local_vertices; lv += grid.num_threads()) {
        if (s_out[lv] == 0.) local_dangling += p_cur[lv];
    }
    local_dangling = cg::reduce(tile32, local_dangling, cg::plus<weight_t>());
    if (tile32.thread_rank() == 0) atomicAdd(pr_reduce + 0, local_dangling);
    grid.sync();

    if (block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, pr_reduce, pr_reduce, 1);
    }
    grid.sync();

    double dangling_mass = pr_reduce[0];

    // Phase 2: compute new p_vis for each local vertex via in-CSR
    weight_t local_l1 = 0.;
    int tile32_num_grid = grid.num_threads() / 32;
    int tile32_id_grid  = (block.num_threads() * block.group_index().x + block.thread_index().x) / 32;

    for (vertex_t lv = tile32_id_grid; lv < local_vertices; lv += tile32_num_grid) {
        weight_t acc = 0.;
        edge_t e_start = in_offset[lv];
        edge_t e_end   = in_offset[lv + 1];
        for (edge_t e = e_start + tile32.thread_rank(); e < e_end; e += 32) {
            vertex_t u_id = in_edge[e];   // global source vertex (modified to local by locate_vertex_pe)
            int u_pe;
            locate_vertex_pe(u_pe, u_id, part_vertex_offset, n_pes);
            // After locate_vertex_pe, u_id is the LOCAL index on u_pe

            weight_t pu = (u_pe == my_pe) ? p_cur[u_id] : nvshmem_double_g(p_cur + u_id, u_pe);
            weight_t su = (u_pe == my_pe) ? s_out[u_id] : nvshmem_double_g(s_out + u_id, u_pe);
            if (su > 0.) acc += (weight_t)((double)in_edge_weight[e] / (double)su * (double)pu);
        }
        acc = cg::reduce(tile32, acc, cg::plus<weight_t>());
        if (tile32.thread_rank() == 0) {
            weight_t new_p = (weight_t)(tau * inv_N + (1.0 - tau) * ((double)acc + dangling_mass * inv_N));
            local_l1 += (new_p > p_cur[lv] ? new_p - p_cur[lv] : p_cur[lv] - new_p);
            p_new[lv] = new_p;
        }
    }

    local_l1 = cg::reduce(tile32, local_l1, cg::plus<weight_t>());
    if (tile32.thread_rank() == 0) atomicAdd(pr_reduce + 1, local_l1);
    grid.sync();

    if (block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, pr_reduce + 1, pr_reduce + 1, 1);
    }
    grid.sync();
}

namespace louvain {

// Run PageRank power iteration until convergence or max_iter, replacing
// p_cur (== private_device_vertex_weight) with the ergodic visit distribution.
// Only executed in directed mode (tau > 0 && gpuGraph->is_directed_()).
static void launch_pagerank(
    vertex_t local_vertices,
    vertex_t total_vertices,
    double tau,
    weight_t *p_cur,               // NVSHMEM: private_device_vertex_weight
    weight_t *p_new,               // plain cudaMalloc scratch (local only)
    vertex_t *in_offset,
    edge_t   *in_edge,
    weight_t *in_edge_weight,
    weight_t *s_out,               // NVSHMEM: private_device_s_out
    vertex_t *part_vertex_offset,
    weight_t *pr_reduce,           // NVSHMEM: 2-element scratch
    int my_pe,
    int n_pes,
    int block_dims,
    size_t d_shared_mem,
    cudaStream_t stream,
    int max_pr_iter = 100,
    double pr_eps   = 1e-6
) {
    int grid_size = 0;
    double inv_N = 1.0 / (double)total_vertices;

    void *args[] = {
        (void *) &local_vertices, (void *) &tau, (void *) &inv_N,
        (void *) &p_cur, (void *) &p_new,
        (void *) &in_offset, (void *) &in_edge, (void *) &in_edge_weight,
        (void *) &s_out, (void *) &part_vertex_offset,
        (void *) &my_pe, (void *) &n_pes, (void *) &pr_reduce
    };

    for (int iter = 0; iter < max_pr_iter; iter++) {
        NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)pagerank_step, block_dims, args, d_shared_mem, &grid_size));
        nvshmem_barrier_all();
        NVSHMEM_CHECK(nvshmemx_collective_launch((void *)pagerank_step, grid_size, block_dims, args, d_shared_mem, stream));
        nvshmemx_barrier_all_on_stream(stream);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));

        // read L1 convergence norm from pr_reduce[1]
        weight_t l1;
        CUDA_RT_CALL(cudaMemcpy(&l1, pr_reduce + 1, sizeof(weight_t), cudaMemcpyDeviceToHost));

        // swap p_new → p_cur for all local vertices
        // (reuse a simple copy kernel; copy from p_new → p_cur)
        copy<weight_t><<<80, 1024, 0, stream>>>(p_new, p_cur, local_vertices);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        // update the args pointer for p_cur (no-op: same pointer, contents updated in-place)

        if (l1 < (weight_t)pr_eps) break;
    }
}

// Launch the three map-equation kernels that score a partition: compute the
// per-module exit weights (q_out) for `community_ids`, then assemble the
// codelength, writing score = -L into Q and the raw exit sum into Q_sum.
// `q_out_delta` is total_vertices doubles of scratch (the community delta array
// is reused between loop iterations).
static void launch_compute_codelength(
    weight_t mass,
    vertex_t local_vertices,
    vertex_t total_vertices,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_q_out,
    weight_t* q_out_delta,
    weight_t* private_device_vertex_weight,
    weight_t* cl_reduce,
    weight_t* Q,
    weight_t* Q_sum,
    int my_pe,
    int n_pes,
    int block_dims,
    size_t d_shared_mem,
    cudaStream_t default_stream,
    // directed-mode extras (nullptr = undirected)
    weight_t* s_out = nullptr,
    double tau = 0.0)
{
    int grid_size = 0;

    // 1) per-PE partial exit weights into q_out_delta (indexed by global module)
    if (s_out != nullptr) {
        // directed: compute q_walk_out using p_vis-weighted out-CSR
        void *a1d[] = {
            (void *) &local_vertices, (void *) &total_vertices,
            (void *) &private_device_offset, (void *) &private_device_edge,
            (void *) &private_device_edge_weight,
            (void *) &private_device_vertex_weight, (void *) &s_out,
            (void *) &private_device_part_vertex_offset,
            (void *) &community_ids, (void *) &q_out_delta,
            (void *) &my_pe, (void *) &n_pes
        };
        NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_q_walk_out_local_atomic, block_dims, a1d, d_shared_mem, &grid_size));
        nvshmem_barrier_all();
        NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_q_walk_out_local_atomic, grid_size, block_dims, a1d, d_shared_mem, default_stream));
    } else {
        void *a1[] = {
            (void *) &local_vertices, (void *) &total_vertices,
            (void *) &private_device_offset, (void *) &private_device_edge,
            (void *) &private_device_edge_weight, (void *) &private_device_part_vertex_offset,
            (void *) &community_ids, (void *) &q_out_delta,
            (void *) &my_pe, (void *) &n_pes
        };
        NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_q_out_local_atomic, block_dims, a1, d_shared_mem, &grid_size));
        nvshmem_barrier_all();
        NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_q_out_local_atomic, grid_size, block_dims, a1, d_shared_mem, default_stream));
    }
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    // 2) reduce partials across PEs into the owning PE's q_out
    void *a2[] = {
            (void *) &local_vertices, (void *) &total_vertices,
            (void *) &shared_device_community_q_out, (void *) &q_out_delta,
            (void *) &private_device_part_vertex_offset, (void *) &my_pe, (void *) &n_pes
    };
    NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_q_out, block_dims, a2, d_shared_mem, &grid_size));
    nvshmem_barrier_all();
    NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_q_out, grid_size, block_dims, a2, d_shared_mem, default_stream));
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    // 2b) directed: apply teleportation correction q_walk_out → q_out
    if (s_out != nullptr) {
        weight_t tau_f = (weight_t)tau;
        apply_teleportation_q_out<<<80, 1024, 0, default_stream>>>(
            local_vertices, tau_f, shared_device_community_weight, shared_device_community_q_out);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
    }

    // 3) assemble codelength -> Q (= -L), Q_sum (= raw Sum_i cut_i)
    void *a3[] = {
            (void *) &mass, (void *) &local_vertices,
            (void *) &shared_device_community_weight, (void *) &shared_device_community_q_out,
            (void *) &private_device_vertex_weight, (void *) &cl_reduce,
            (void *) &my_pe, (void *) &n_pes, (void *) &Q, (void *) &Q_sum
    };
    NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_codelength, block_dims, a3, d_shared_mem, &grid_size));
    nvshmem_barrier_all();
    NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_codelength, grid_size, block_dims, a3, d_shared_mem, default_stream));
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}
}  // namespace louvain

void louvain::run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter,
                           const int max_phases, const double tau) {
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();

    // stream create
    cudaStream_t default_stream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&default_stream, cudaStreamDefault));

    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    nvshmem_barrier_all();

    auto mass = gpuGraph->get_mass_();
    auto local_vertices = gpuGraph->get_local_vertices_();
    auto local_edges = gpuGraph->get_local_edges_();
    auto total_vertices = gpuGraph->get_total_vertices_();
    auto *part_vertex_offset = gpuGraph->get_part_vertex_offset_();
    weight_t Q_host = 0;
    weight_t Q_old_host = -1;

    // private memory
    auto *private_device_offset = gpuGraph->get_private_device_offset_();
    auto *private_device_edge = gpuGraph->get_private_device_edge_();
    auto *private_device_edge_weight = gpuGraph->get_private_device_edge_weight_();
    auto *private_device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    auto *private_device_vertex_weight = gpuGraph->get_private_device_vertex_weight_();

    // init bins
    BIN* bins = new BIN(BIN_NUM, local_vertices);

    // directed-mode PageRank scratch
    const bool directed = gpuGraph->is_directed_();
    weight_t *pr_p_cur   = nullptr;  // NVSHMEM: symmetric current p (remote PEs read it)
    weight_t *pr_p_new   = nullptr;  // local-only scratch for PageRank new p
    weight_t *pr_reduce  = nullptr;  // NVSHMEM: 2 doubles [dangling_mass, l1_norm]
    vertex_t *pr_in_offset = nullptr;
    edge_t   *pr_in_edge   = nullptr;
    weight_t *pr_in_edge_weight = nullptr;
    weight_t *pr_s_out   = nullptr;
    if (directed) {
        // pr_p_cur MUST be symmetric: pagerank_step fetches it from remote PEs via
        // nvshmem_double_g. private_device_vertex_weight is a plain cudaMalloc and
        // cannot be used as the symmetric source (illegal access at -np >= 2).
        pr_p_cur      = (weight_t *) nvshmem_malloc(total_vertices * sizeof(weight_t));
        CUDA_RT_CALL(cudaMemset(pr_p_cur, 0, total_vertices * sizeof(weight_t)));
        CUDA_RT_CALL(cudaMalloc((void **) &pr_p_new, sizeof(weight_t) * local_vertices));
        pr_reduce     = (weight_t *) nvshmem_malloc(2 * sizeof(weight_t));
        pr_in_offset  = gpuGraph->get_private_device_in_offset_();
        pr_in_edge    = gpuGraph->get_private_device_in_edge_();
        pr_in_edge_weight = gpuGraph->get_private_device_in_edge_weight_();
        pr_s_out      = gpuGraph->get_private_device_s_out_();
    }

    // nvshmem shared memory
    auto *shared_device_community_weight = gpuGraph->get_shared_device_community_weight_();
    auto *shared_device_community_delta_weight = gpuGraph->get_shared_device_community_delta_weight_();
    auto *shared_device_community_ids = gpuGraph->get_shared_device_community_ids_();
    auto *shared_device_community_ids_new = gpuGraph->get_shared_device_community_ids_new_();
    auto *shared_device_community_q_out = gpuGraph->get_shared_device_community_q_out_();
    auto *Q = (weight_t *) nvshmem_malloc(sizeof(weight_t));        // score = -L (maximised)
    CUDA_RT_CALL(cudaMemset(Q, 0, sizeof(weight_t)));
    auto *Q_sum = (weight_t *) nvshmem_malloc(sizeof(weight_t));    // raw Sum_i cut_i, read by gain kernels
    CUDA_RT_CALL(cudaMemset(Q_sum, 0, sizeof(weight_t)));
    auto *cl_reduce = (weight_t *) nvshmem_malloc(4 * sizeof(weight_t));   // codelength reduction scratch
    CUDA_RT_CALL(cudaMemset(cl_reduce, 0, 4 * sizeof(weight_t)));

    // --- Reject-staleness fix -------------------------------------------------
    // community_weight / community_q_out / Q_sum are updated IN PLACE for the
    // proposed partition (ids_new) every iteration (step b + step c). On a
    // *rejected* iteration ids_new is rolled back to community_ids, but those
    // three arrays are left holding the REJECTED proposal's values. The next
    // iteration's move-gain kernels then read state that is inconsistent with
    // community_ids, and the incremental step b accumulates from a wrong base
    // (this is the other half of the bug commit cc7b73c only half-fixed by
    // rolling back ids_new). We keep a backup of the state that is consistent
    // with the last *accepted* partition and restore it on rejection.
    const vertex_t max_total_vertices = total_vertices;   // original (largest) level
    weight_t *cw_backup   = nullptr;   // community_weight  of last accepted partition
    weight_t *cq_backup   = nullptr;   // community_q_out   of last accepted partition
    weight_t *qsum_backup = nullptr;   // Q_sum (raw Sum_i cut) of last accepted partition
    CUDA_RT_CALL(cudaMalloc((void **) &cw_backup,   max_total_vertices * sizeof(weight_t)));
    CUDA_RT_CALL(cudaMalloc((void **) &cq_backup,   max_total_vertices * sizeof(weight_t)));
    CUDA_RT_CALL(cudaMalloc((void **) &qsum_backup, sizeof(weight_t)));

    int block_dims = 1024;
    int grid_size = 0;
    size_t d_shared_mem = 0;

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    double start;
    double stop;
    double start_total;
    double stop_total;
    double start_phase;
    double stop_phase;
    double start_in_loop;
    double stop_in_loop;

    double coarsen_graph_total_time = 0.;
    double main_loop_total_time = 0.;
    double symbolic_time = 0.;
    double numeric_time = 0.;
    double update_community_time = 0.;
    double update_weight_time = 0.;
    double compute_modularity_time = 0.;

    int phase_num = 0;
    int loop_total = 0;
    // Track the original mass for restoring after directed Phase 0.
    weight_t original_mass = mass;
    start_total = MPI_Wtime();
    while (phase_num < max_phases && (Q_host - Q_old_host) > threshold) {

        // Directed mode is only fully supported in Phase 0 (before coarsening).
        // Phase 1+ uses the coarsened out-CSR (undirected symmetrized), so the in-CSR
        // would be stale. Fall back to undirected coarsening path for Phase 1+.
        // B6 (full directed coarsening) is future work.
        bool phase_directed = directed && (phase_num == 0);

        weight_t new_Q;
        weight_t cur_Q;
        weight_t phase_initial_score;   // codelength score (-L) at the start of this phase
        vertex_t begin_vertex_id = part_vertex_offset[my_pe];
        vertex_t end_vertex_id = part_vertex_offset[my_pe + 1];

        int loop_num = 0;
        double loop_time = 0.;

        start_phase = MPI_Wtime();
        if (my_pe == 0) {
            printf("----------------------------------------\n");
            printf("Phase %d\n", phase_num);
        }

        // 1. Init community id for new graph

        init_community_id<<<80, 1024, 0, default_stream>>>(shared_device_community_ids, begin_vertex_id, end_vertex_id, local_vertices);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        // 2. Compute the vertices and communities weights

        reduce_vertices_weights<<<80, 1024, 0, default_stream>>>(local_vertices,
                                                                 private_device_offset,
                                                                 private_device_edge_weight,
                                                                 private_device_vertex_weight);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        // 2b. Directed mode: replace k_v with PageRank ergodic visit probability p_vis[v].
        //     Normalize k_v to probability units first (uniform init), then iterate.
        if (phase_directed) {
            // Initialize p_vis = k_v / Σk_v  (out-strength as initial distribution)
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            weight_t local_sum = thrust::reduce(
                thrust::device_pointer_cast(private_device_vertex_weight),
                thrust::device_pointer_cast(private_device_vertex_weight) + local_vertices,
                (weight_t)0.0, thrust::plus<weight_t>());
            // Global sum via MPI (avoids NVSHMEM host-side reduce which may not be available)
            weight_t global_sum = 0.0;
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if (global_sum > 0.) {
                weight_t inv_sum = (weight_t)(1.0 / (double)global_sum);
                // Scale private_device_vertex_weight by inv_sum in-place (no __device__ lambda)
                thrust::transform(
                    thrust::device_pointer_cast(private_device_vertex_weight),
                    thrust::device_pointer_cast(private_device_vertex_weight) + local_vertices,
                    thrust::make_constant_iterator(inv_sum),
                    thrust::device_pointer_cast(private_device_vertex_weight),
                    thrust::multiplies<weight_t>());
            }
            // Stage the normalized init into the symmetric buffer pr_p_cur, run
            // PageRank there (so remote PEs can read each other's p via NVSHMEM),
            // then copy the converged p_vis back into private_device_vertex_weight.
            // copy(src, dst, len): dst[i] = src[i].
            copy<weight_t><<<80, 1024, 0, default_stream>>>(private_device_vertex_weight, pr_p_cur, local_vertices);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            // Run PageRank power iteration on the symmetric buffer
            launch_pagerank(local_vertices, total_vertices, tau,
                            pr_p_cur, pr_p_new,
                            pr_in_offset, pr_in_edge, pr_in_edge_weight, pr_s_out,
                            private_device_part_vertex_offset,
                            pr_reduce, my_pe, n_pes,
                            block_dims, d_shared_mem, default_stream);
            copy<weight_t><<<80, 1024, 0, default_stream>>>(pr_p_cur, private_device_vertex_weight, local_vertices);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            // After convergence: p_vis is in private_device_vertex_weight (normalized, Σ = 1)
            // Override mass to 1.0 so codelength uses probability units.
            mass = 1.0;
        } else if (phase_num > 0) {
            // Phase 1+: restore mass from the coarsened graph (out-strength sum, not p_vis units).
            mass = original_mass;
        }

        // Seed community weight from vertex weight for the singleton partition
        // (community[v] = v at the start of each phase, so vol[i] = pvw[i_local]).
        // The inner move-loop updates this incrementally after each reassignment.
        // NOTE: copy(src, dst, len) does dst[i] = src[i]; here dst = community_weight.
        copy<weight_t><<<80, 1024, 0, default_stream>>>(private_device_vertex_weight, shared_device_community_weight, local_vertices);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        // 3. Compute the initial codelength score (-L) of this phase
        launch_compute_codelength(mass, local_vertices, total_vertices,
                                  private_device_offset, private_device_edge, private_device_edge_weight,
                                  private_device_part_vertex_offset, shared_device_community_ids,
                                  shared_device_community_weight, shared_device_community_q_out,
                                  shared_device_community_delta_weight, private_device_vertex_weight,
                                  cl_reduce, Q, Q_sum, my_pe, n_pes, block_dims, d_shared_mem, default_stream,
                                  phase_directed ? pr_s_out : nullptr, phase_directed ? tau : 0.0);

        CUDA_RT_CALL(cudaMemcpy(&new_Q, Q, sizeof(weight_t) , cudaMemcpyDeviceToHost));

        cur_Q = new_Q - 1;
        phase_initial_score = new_Q;

        // Seed the accepted-state backups with this phase's initial (singleton)
        // partition state, which the initial codelength call above just produced.
        CUDA_RT_CALL(cudaMemcpyAsync(cw_backup, shared_device_community_weight,
                                     local_vertices * sizeof(weight_t),
                                     cudaMemcpyDeviceToDevice, default_stream));
        CUDA_RT_CALL(cudaMemcpyAsync(cq_backup, shared_device_community_q_out,
                                     local_vertices * sizeof(weight_t),
                                     cudaMemcpyDeviceToDevice, default_stream));
        CUDA_RT_CALL(cudaMemcpyAsync(qsum_backup, Q_sum, sizeof(weight_t),
                                     cudaMemcpyDeviceToDevice, default_stream));
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        if(my_pe == 0) {
            printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n", "Loop", "L(bits)", "dL", "time(s)", "time(ms)");
            printf("|------------|------------|------------|------------|------------|\n");
            printf("| %-10d | %-10f | %-10f | %-10f | %-10f |\n", 0, -new_Q, 0., 0., 0.);
        }

        bool up_down = true;
        // Exit only after two consecutive non-improving iterations (one up_down=true, one
        // up_down=false). Without this, directed graphs with tau≈0.15 exit after the first
        // up_down=true pass because beneficial merges go to higher-ID communities — the
        // up_down=false pass is never reached and the partition stays at the identity.
        int consec_no_improve = 0;

        bins->bin_create(private_device_offset, local_vertices);


        // 4. Update the community id of each vertex (main loop)
        // The presence of negative modularity leads to a direct termination of the cycle.
        // The `loop_num < max_iter` bound is a hard safety cap: it stops the loop from
        // spinning forever if the objective fails to converge below `threshold` (which
        // happened in directed mode when q_out was mis-derived and L drifted negative).
        while (consec_no_improve < 2 && loop_num < max_iter) {

            cur_Q = new_Q;

            start = MPI_Wtime();
            start_in_loop = MPI_Wtime();

            // a) update community id
            calculate_eicj_and_move_vertex_bin(local_vertices,
                                                begin_vertex_id,
                                                private_device_offset,
                                                private_device_edge,
                                                private_device_edge_weight,
                                                private_device_vertex_weight,
                                                private_device_part_vertex_offset,
                                                shared_device_community_ids_new,
                                                shared_device_community_ids,
                                                shared_device_community_weight,
                                                shared_device_community_q_out,
                                                Q_sum,
                                                mass,
                                                my_pe,
                                                n_pes,
                                                up_down,
                                                bins,
                                                streams,
                                               total_vertices,
                                               default_stream,
                                               phase_directed ? pr_s_out : nullptr,
                                               phase_directed ? tau : 0.0,
                                               phase_directed ? pr_in_offset : nullptr,
                                               phase_directed ? pr_in_edge : nullptr,
                                               phase_directed ? pr_in_edge_weight : nullptr);



            up_down = !up_down;

            nvshmem_barrier_all();
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            stop_in_loop = MPI_Wtime();
            update_community_time += (stop_in_loop - start_in_loop);
            start_in_loop = MPI_Wtime();

            // b) update community weight
            void *kernel_args_ccw[] = {
                    (void *) &local_vertices,
                    (void *) &total_vertices,
                    (void *) &private_device_vertex_weight,
                    (void *) &shared_device_community_ids,
                    (void *) &shared_device_community_ids_new,
                    (void *) &shared_device_community_weight,
                    (void *) &shared_device_community_delta_weight,
                    (void *) &private_device_part_vertex_offset,
                    (void *) &my_pe,
                    (void *) &n_pes
            };
            NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight_local_atomic, block_dims, kernel_args_ccw, d_shared_mem, &grid_size));
            nvshmem_barrier_all();
            NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight_local_atomic, grid_size, block_dims, kernel_args_ccw, d_shared_mem, default_stream));
            nvshmemx_barrier_all_on_stream(default_stream);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight, block_dims, kernel_args_ccw, d_shared_mem, &grid_size));
            nvshmem_barrier_all();
            NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight, grid_size, block_dims, kernel_args_ccw, d_shared_mem, default_stream));
            nvshmemx_barrier_all_on_stream(default_stream);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            stop_in_loop = MPI_Wtime();
            update_weight_time += (stop_in_loop - start_in_loop);
            start_in_loop = MPI_Wtime();

            // c) compute the new codelength score (-L) for the proposed partition
            launch_compute_codelength(mass, local_vertices, total_vertices,
                                      private_device_offset, private_device_edge, private_device_edge_weight,
                                      private_device_part_vertex_offset, shared_device_community_ids_new,
                                      shared_device_community_weight, shared_device_community_q_out,
                                      shared_device_community_delta_weight, private_device_vertex_weight,
                                      cl_reduce, Q, Q_sum, my_pe, n_pes, block_dims, d_shared_mem, default_stream,
                                      phase_directed ? pr_s_out : nullptr, phase_directed ? tau : 0.0);

            stop_in_loop = MPI_Wtime();
            compute_modularity_time += (stop_in_loop - start_in_loop);

            CUDA_RT_CALL(cudaMemcpy(&new_Q, Q, sizeof(weight_t) , cudaMemcpyDeviceToHost));

            if ((new_Q - cur_Q) > threshold) {
                copy<vertex_t><<<128, 1024, 0, default_stream>>>(shared_device_community_ids_new,
                                                                 shared_device_community_ids,
                                                                 local_vertices);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
                consec_no_improve = 0;
                // Accepted: the in-place community_weight/q_out/Q_sum now describe the
                // new accepted partition (community_ids). Refresh the backups.
                CUDA_RT_CALL(cudaMemcpyAsync(cw_backup, shared_device_community_weight,
                                             local_vertices * sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaMemcpyAsync(cq_backup, shared_device_community_q_out,
                                             local_vertices * sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaMemcpyAsync(qsum_backup, Q_sum, sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            } else {
                new_Q = cur_Q;
                consec_no_improve++;
                // Rollback community_ids_new to the last accepted partition. Without this,
                // the next iteration's step b) would apply the rejected partition's delta
                // a second time, corrupting community_weight (it would double each rejection).
                copy<vertex_t><<<128, 1024, 0, default_stream>>>(shared_device_community_ids,
                                                                  shared_device_community_ids_new,
                                                                  local_vertices);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
                // Rejected: community_weight/q_out/Q_sum still hold the REJECTED proposal's
                // values. Restore the state consistent with community_ids so the next
                // iteration's move-gains (and incremental step b) start from the right base.
                CUDA_RT_CALL(cudaMemcpyAsync(shared_device_community_weight, cw_backup,
                                             local_vertices * sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaMemcpyAsync(shared_device_community_q_out, cq_backup,
                                             local_vertices * sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaMemcpyAsync(Q_sum, qsum_backup, sizeof(weight_t),
                                             cudaMemcpyDeviceToDevice, default_stream));
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            }

            loop_num++;
            stop = MPI_Wtime();
            if(my_pe == 0){
                printf("| %-10d | %-10f | %-10f | %-10f | %-10f |\n", loop_num, -new_Q, -(new_Q - cur_Q), (stop - start), (stop - start) * 1000);
            }
            loop_time += (stop - start) * 1000;
            nvshmem_barrier_all();
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        }

        nvshmem_barrier_all();
        CUDA_RT_CALL(cudaDeviceSynchronize());

        loop_time /= loop_num;
        loop_total += loop_num;
        stop_phase = MPI_Wtime();
        main_loop_total_time += (stop_phase - start_phase);
        if (my_pe == 0) {
            printf("Main loop total execution time: %f s, %f ms\n", (stop_phase - start_phase), double((stop_phase - start_phase) * 1000));
            printf("The average execution time per loop: %f s, %f ms\n", loop_time / 1000, loop_time);
        }

        // The map-equation codelength is not invariant across coarsening (the
        // per-node entropy term changes level to level), so phase continuation is
        // driven by this phase's own improvement (final vs. initial score) rather
        // than by comparing scores across levels. For modularity these are equal.
        Q_old_host = phase_initial_score;
        Q_host = new_Q;
        if ((Q_host - Q_old_host) <= threshold) break;

        start_phase = MPI_Wtime();

        coarsen_graph_mg::coarsen_graph(hostGraph, gpuGraph, my_pe, n_pes, default_stream, streams, symbolic_time, numeric_time);

        stop_phase = MPI_Wtime();
        coarsen_graph_total_time += (stop_phase - start_phase);
        phase_num++;
        if (my_pe == 0) {
            printf("Coarsen graph execution time: %f s, %f ms\n", (stop_phase - start_phase), double((stop_phase - start_phase) * 1000));
        }

        local_vertices = gpuGraph->get_local_vertices_();
        total_vertices = gpuGraph->get_total_vertices_();

        // In directed mode, the in-CSR is NOT rebuilt by coarsen_graph (B6 is not yet implemented).
        // The directed path is only active for Phase 0; Phase 1+ uses undirected coarsened graph.
        // Reallocate p_new scratch for the new (smaller) local vertex count.
        if (directed) {
            CUDA_RT_CALL(cudaFree(pr_p_new));
            CUDA_RT_CALL(cudaMalloc((void **) &pr_p_new, sizeof(weight_t) * local_vertices));
        }

        nvshmem_barrier_all();
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    nvshmem_barrier_all();
    CUDA_RT_CALL(cudaDeviceSynchronize());

    stop_total = MPI_Wtime();
    if (my_pe == 0) {
        printf("----------------------------------------\n");
        printf("Total time for clustering   : %f s\n", main_loop_total_time);
        printf("Total time for coarsen graph: %f s\n", coarsen_graph_total_time);
        printf("TOTAL TIME                  : %f s\n", (stop_total - start_total));
//        printf("Total time for updating community: %f s\n", update_community_time);
//        printf("Total time for updating weight: %f s\n", update_weight_time);
//        printf("Total time for computing modularity: %f s\n", compute_modularity_time);
//        printf("Total time for symb. phase: %f s\n", symbolic_time);
//        printf("Total time for num. phase: %f s\n", numeric_time);
//        printf("Total phases: %d\n", phase_num++);
//        printf("Total loops: %d\n", loop_total);
    }
    CUDA_RT_CALL(cudaFree(cw_backup));
    CUDA_RT_CALL(cudaFree(cq_backup));
    CUDA_RT_CALL(cudaFree(qsum_backup));
    nvshmem_free(Q);
    nvshmem_free(Q_sum);
    nvshmem_free(cl_reduce);
    if (directed) {
        CUDA_RT_CALL(cudaFree(pr_p_new));
        nvshmem_free(pr_reduce);
        nvshmem_free(pr_p_cur);
    }
}

