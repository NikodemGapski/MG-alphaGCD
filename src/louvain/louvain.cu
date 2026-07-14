#include "../../include/louvain/louvain.cuh"
#include "../../include/bin/BIN.cuh"

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

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

// Directed map equation move gain under Infomap's UNRECORDED teleportation:
//   q_out[i] = τ·(N - n_i)/N·q_vis[i] + (1-τ)·q_walk_out[i]
// consistent with apply_teleportation_q_out. n_i is the number of ORIGINAL nodes in module i
// and N the total: a walker in i teleports to a uniformly random node, which lands outside i
// with probability (N - n_i)/N.
//
// The previous model was *proportional* teleportation, τ·q_vis·(1-q_vis). That is not what
// infomap optimises, and it is minimised by lumping the graph into a couple of giant modules
// (on wiki-Vote its best partition is two ~2200-node blobs holding 29% of the flow).
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
    weight_t in_n_flow = 0.,   // walk in-flow into v from n
    // Total normalised out-flow of v, EXCLUDING self-loops:  sum_{w != v} w(v,w)/s_out[v].
    // The old code hard-coded this to 1, which is only true at level 0 for a vertex that has
    // out-edges. It is WRONG for:
    //   * dangling vertices (s_out == 0): true walk-out flow is 0, but 1-em charges a full p.
    //     wiki-Vote has 2187 of them.
    //   * every coarse super-node under B6: s_out(C) := p_C while sum_D f(C,D) = p_C minus the
    //     dangling mass inside C, so the total out-fraction is < 1.
    //   * self-loops f(C,C) (which every coarse graph has): that flow stays with C wherever C
    //     goes, so it must not count as leaving either module.
    // With dangling nodes and self-loops present the 1-em form is off by up to 0.43 bits and
    // disagrees with the exact dL on the SIGN of the move 8% of the time
    // (tools/directed_gain_dangling_check.py); passing the real value is exact to ~2e-15.
    weight_t tot_out = 1.,
    // Unrecorded teleportation needs the module NODE COUNTS (in original nodes, so they must
    // be carried through coarsening like the flow) and the total N.
    weight_t n_m = 0.,     // n_i of the old module m, INCLUDING v
    weight_t n_n = 0.,     // n_i of the candidate module n, excluding v
    weight_t nc_v = 0.,    // how many original nodes v itself represents (1 at level 0)
    weight_t N_total = 1.,
    // DANGLING flow. A node with no out-links has nothing to follow, so it teleports with
    // probability ONE, not tau -- its exit flow is (N-n_i)/N * p, coefficient 1. Charging it
    // only tau makes a module packed with dangling nodes almost free to sit in, which is why a
    // partition of two ~2200-node blobs beat infomap's on wiki-Vote (26% of it is dangling).
    // Omitting these terms puts the WRONG SIGN on 23% of moves
    // (tools/directed_gain_dangling_check.py); with them the gain is exact to ~2e-15.
    weight_t d_m = 0.,     // dangling flow of module m, INCLUDING v's
    weight_t d_n = 0.,     // dangling flow of module n, excluding v's
    weight_t d_v = 0.      // dangling flow of v itself (== p_v at level 0 if v is dangling)
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

    // Exact deltas for q_out[i] = τ·(N-n_i)/N·q_vis[i] + (1-τ)·q_walk_out[i] when v moves m->n.
    // The walk flow that leaves m is p·(tot_out - em), NOT p·(1 - em): see tot_out above.
    // The teleport factor changes too, because n_m shrinks by nc_v and n_n grows by nc_v.
    const double to = (double)tot_out;
    const double N  = (double)N_total;
    const double am  = (N - (double)n_m) / N;               // teleport factor of m, before
    const double am2 = (N - (double)n_m + (double)nc_v) / N; // ... after v leaves
    const double an  = (N - (double)n_n) / N;               // teleport factor of n, before
    const double an2 = (N - (double)n_n - (double)nc_v) / N; // ... after v joins
    const double dm = (double)d_m, dn = (double)d_n, dv = (double)d_v;
    // q_out[i] = (N-n_i)/N * [ tau*q_vis[i] + (1-tau)*d_i ] + (1-tau)*q_walk_out[i]
    const double dqm = am2 * (tv * (qm - p) + (1.0-tv) * (dm - dv))
                     - am  * (tv *  qm      + (1.0-tv) *  dm)
                     + (1.0-tv) * (im - p * (to - em));
    const double dqn = an2 * (tv * (qn + p) + (1.0-tv) * (dn + dv))
                     - an  * (tv *  qn      + (1.0-tv) *  dn)
                     + (1.0-tv) * (p * (to - en) - in_);

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

// Directed map equation: convert q_walk_out → q_out using Infomap's UNRECORDED teleportation.
//   q_out[i] = τ·(N - n_i)/N·q_vis[i] + (1-τ)·q_walk_out[i]
// A walker in module i teleports (probability τ) to a uniformly random node, which lands
// OUTSIDE i with probability (N - n_i)/N -- so n_i is a count of ORIGINAL nodes and must be
// carried through coarsening, exactly like the flow (see aggregate_coarse_flow).
//
// This replaces the previous *proportional* model τ·q_vis·(1-q_vis), which is not what infomap
// optimises and is minimised by lumping the graph into a couple of giant modules.
__global__ void __launch_bounds__(1024, 1)
apply_teleportation_q_out(
    vertex_t local_vertices,
    weight_t tau,
    weight_t* shared_device_community_weight,      // q_vis[i]
    weight_t* shared_device_community_q_out,       // in: q_walk_out[i], out: q_out[i]
    weight_t* shared_device_community_node_count,     // n_i, in ORIGINAL nodes
    weight_t* shared_device_community_dangling_flow,  // d_i: flow on DANGLING nodes of i
    weight_t N_total
)
{
    for (vertex_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < local_vertices; i += blockDim.x * gridDim.x) {
        weight_t q_vis  = shared_device_community_weight[i];
        weight_t q_walk = shared_device_community_q_out[i];
        weight_t n_i    = shared_device_community_node_count[i];
        weight_t d_i    = shared_device_community_dangling_flow[i];
        const double tele = ((double)N_total - (double)n_i) / (double)N_total;
        // dangling nodes teleport with probability 1, everyone else with probability tau
        shared_device_community_q_out[i] = (weight_t)(
            tele * (tau * (double)q_vis + (1.0 - tau) * (double)d_i)
            + (1.0 - tau) * (double)q_walk);
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
    weight_t* Q_sum,
    // Sum_v F(p_vis_v) of the ORIGINAL (level-0) vertices. nullptr = derive it from this
    // level's vertex weights. The node-entropy term is a constant within a phase, but it is
    // re-based by coarsening (super-nodes have larger p than the nodes they contain), so
    // deriving it per level makes L incomparable across phases -- see the phase-continuity
    // check in louvain::run. Passing the level-0 value keeps L the TRUE codelength.
    weight_t* s_node_level0
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
    if (s_node_level0 == nullptr) {
        for (vertex_t v = grid.thread_rank(); v < local_vertices; v += grid.num_threads()) {
            s_node += plogp(private_device_vertex_weight[v] / mass);
        }
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
        weight_t q_raw  = cl_reduce[0];
        weight_t Qp     = q_raw / mass;
        weight_t s_node_total = (s_node_level0 != nullptr) ? s_node_level0[0] : cl_reduce[3];
        weight_t L      = plogp(Qp) - 2.0 * cl_reduce[1] - s_node_total + cl_reduce[2];
        // Publish the node-entropy term so Phase 0 can capture it as the level-0 constant.
        cl_reduce[3]    = s_node_total;
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

// ---------------------------------------------------------------------------
// Clean local-moving kernel: ONE WARP PER VERTEX.
//
// The binned tile/block kernels above compute the right thing *on paper* --
// tools/kernel_emulator.py reimplements their exact logic (bin capacities, the
// (c*107)%size hash with linear probing, the per-lane slot scan, the tree reduction and
// its ID tie-break, the up_down gate) and reproduces the CPU reference: from singletons
// wiki-Vote reaches L=11.433430 / 1224 communities after pass 1. The GPU instead reaches
// L=12.268406 and then collapses to the all-in-one basin, so it is diverging from its own
// source semantics at runtime (a race / UB), not computing a different algorithm.
//
// This kernel deliberately removes every mechanism that could be responsible:
//   * one warp per vertex -- no sub-warp cooperative tiles,
//   * hash table in GLOBAL memory, slice [2*offset[v], 2*offset[v+1]) -- capacity 2*deg(v),
//     so it is at most half full and probing can never wrap into another vertex's slots,
//   * no reuse of the hash storage as reduction scratch,
//   * a single launch on one stream -- no 10 concurrent per-bin kernels.
// It is otherwise semantically identical to the binned path (same gain, same tie-breaks,
// same up_down gate), so the two can be A/B'd with -move.
__global__ void __launch_bounds__(256, 1)
move_vertices_warp(
    vertex_t begin_vertex_id,
    vertex_t local_vertices,
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
    vertex_t* hash_key,        // global scratch, 2 * local_edges entries
    weight_t* hash_val,
    weight_t* hash_in_val,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    weight_t* s_out_local,     // nullptr = undirected
    double tau,
    vertex_t* in_offset,       // directed in-CSR (local-indexed; in_edge holds GLOBAL ids)
    edge_t* in_edge,
    weight_t* in_edge_weight,
    // directed: unrecorded teleportation needs per-vertex and per-module ORIGINAL node counts
    weight_t* node_count,                 // nc[v]: how many original nodes v represents
    weight_t* community_node_count,       // n_i per module
    weight_t* dangling_flow,              // d[v]: flow on dangling originals inside v
    weight_t* community_dangling_flow,    // d_i per module
    weight_t  N_total
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    const bool directed = (s_out_local != nullptr);
    if (!directed) mass = mass / 2.0;   // undirected passes m; move_gain re-doubles it
    const weight_t q_total = Q_sum[0];

    const int warps_in_grid = grid.num_threads() / 32;
    const int warp_id_grid  = grid.thread_rank() / 32;

    for (vertex_t v = warp_id_grid; v < local_vertices; v += warps_in_grid) {
        const edge_t edge_lb = private_device_offset[v];
        const edge_t edge_rb = private_device_offset[v + 1];
        const vertex_t src_community_id = shared_device_community_ids[v];

        // Isolated vertex: no neighbours, so no candidate community. It must stay put.
        if (edge_lb == edge_rb) {
            if (warp.thread_rank() == 0) shared_device_community_ids_new_[v] = src_community_id;
            continue;
        }

        const edge_t hash_lb = edge_lb * 2;
        const edge_t hash_rb = edge_rb * 2;
        const vertex_t hash_size = (vertex_t)(hash_rb - hash_lb);   // = 2*deg(v)

        for (edge_t s = hash_lb + warp.thread_rank(); s < hash_rb; s += 32) {
            hash_key[s]    = UINT32_MAX;
            hash_val[s]    = 0.;
            hash_in_val[s] = 0.;
        }
        warp.sync();

        const weight_t ki = private_device_vertex_weight[v];
        const weight_t sv = directed ? s_out_local[v] : (weight_t)1.0;

        // Gather the weight from v to each neighbouring community (hash), and to its own
        // community (eici). Self-loops are skipped entirely: that flow stays with v wherever v
        // goes, so it never exits any module (level-0 graphs have none, but every COARSE graph
        // does). tot_out is v's total normalised out-flow excluding self-loops -- the directed
        // gain needs the real value, not the 1 it used to assume (see move_gain_directed_approx).
        weight_t eici = 0.;
        weight_t tot_out = 0.;
        for (edge_t e = edge_lb + warp.thread_rank(); e < edge_rb; e += 32) {
            const vertex_t nb_global = private_device_edge[e];
            if (nb_global == v + begin_vertex_id) continue;      // self-loop

            vertex_t neighbor_id = nb_global;
            int pe_dst;
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            const vertex_t dst_community_id = (pe_dst == my_pe)
                ? shared_device_community_ids[neighbor_id]
                : nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            const weight_t w = private_device_edge_weight[e];
            const weight_t w_norm = (directed && sv > 0.) ? w / sv : w;
            tot_out += w_norm;

            if (dst_community_id != src_community_id) {
                vertex_t h = (vertex_t)(((unsigned long long)dst_community_id * 107ULL) % hash_size);
                while (true) {
                    const vertex_t old = atomicCAS(hash_key + hash_lb + h, UINT32_MAX, dst_community_id);
                    if (old == UINT32_MAX || old == dst_community_id) {
                        atomicAdd(hash_val + hash_lb + h, w_norm);
                        break;
                    }
                    h = (h + 1) % hash_size;   // at most half full -> always terminates
                }
            } else {
                eici += w_norm;
            }
        }
        eici    = cg::reduce(warp, eici,    cg::plus<weight_t>());
        tot_out = cg::reduce(warp, tot_out, cg::plus<weight_t>());
        warp.sync();

        // Directed: walk in-flow into v from each module. Look up (never insert) so the
        // table cannot overflow. Remote in-neighbours have no symmetric p[u]; skip them
        // (exact at -np 1). Own-module in-flow is reduced into in_m.
        weight_t in_m = 0.;
        if (directed) {
            for (edge_t e = in_offset[v] + warp.thread_rank(); e < in_offset[v + 1]; e += 32) {
                vertex_t u = in_edge[e];
                if (u == v + begin_vertex_id) continue;   // self-loop: moves with v, never exits
                int pe_u;
                locating_vertex(pe_u, u, private_device_part_vertex_offset, n_pes);
                if (pe_u != my_pe) continue;
                const weight_t su = s_out_local[u];
                if (su <= 0.) continue;
                const weight_t contrib = (weight_t)((double)private_device_vertex_weight[u]
                                                    * (double)in_edge_weight[e] / (double)su);
                const vertex_t cu = shared_device_community_ids[u];
                if (cu == src_community_id) {
                    in_m += contrib;
                } else {
                    vertex_t h = (vertex_t)(((unsigned long long)cu * 107ULL) % hash_size);
                    while (hash_key[hash_lb + h] != UINT32_MAX) {
                        if (hash_key[hash_lb + h] == cu) {
                            atomicAdd(hash_in_val + hash_lb + h, contrib);
                            break;
                        }
                        h = (h + 1) % hash_size;
                    }
                }
            }
            in_m = cg::reduce(warp, in_m, cg::plus<weight_t>());
        }
        warp.sync();

        // community ids are GLOBAL (init_community_id writes the global vertex id), so the
        // owning PE has to be resolved before indexing the symmetric arrays.
        vertex_t src_local = src_community_id;
        int pe_src;
        locating_vertex(pe_src, src_local, private_device_part_vertex_offset, n_pes);
        weight_t aci, qout_m, n_m = 0., d_m = 0.;
        if (pe_src == my_pe) {
            aci    = shared_device_community_weight[src_local];
            qout_m = shared_device_community_q_out[src_local];
            if (directed) {
                n_m = community_node_count[src_local];
                d_m = community_dangling_flow[src_local];
            }
        } else {
            aci    = nvshmem_double_g(shared_device_community_weight + src_local, pe_src);
            qout_m = nvshmem_double_g(shared_device_community_q_out + src_local, pe_src);
            if (directed) {
                n_m = nvshmem_double_g(community_node_count + src_local, pe_src);
                d_m = nvshmem_double_g(community_dangling_flow + src_local, pe_src);
            }
        }
        aci -= ki;   // move_gain expects vol(m) - k_v
        const weight_t nc_v = directed ? node_count[v] : (weight_t)0.;
        const weight_t d_v  = directed ? dangling_flow[v] : (weight_t)0.;

        // Score every candidate community; each lane scans its own stride of slots.
        weight_t best_gain = 0.;                    // only strictly-improving moves are taken
        vertex_t best_dst  = src_community_id;
        for (edge_t s = hash_lb + warp.thread_rank(); s < hash_rb; s += 32) {
            const vertex_t cand = hash_key[s];
            if (cand == UINT32_MAX) continue;

            vertex_t cand_local = cand;
            int pe_c;
            locating_vertex(pe_c, cand_local, private_device_part_vertex_offset, n_pes);
            weight_t acj, qout_n, n_n = 0., d_n = 0.;
            if (pe_c == my_pe) {
                acj    = shared_device_community_weight[cand_local];
                qout_n = shared_device_community_q_out[cand_local];
                if (directed) {
                    n_n = community_node_count[cand_local];
                    d_n = community_dangling_flow[cand_local];
                }
            } else {
                acj    = nvshmem_double_g(shared_device_community_weight + cand_local, pe_c);
                qout_n = nvshmem_double_g(shared_device_community_q_out + cand_local, pe_c);
                if (directed) {
                    n_n = nvshmem_double_g(community_node_count + cand_local, pe_c);
                    d_n = nvshmem_double_g(community_dangling_flow + cand_local, pe_c);
                }
            }

            weight_t g;
            if (directed) {
                g = move_gain_directed_approx(ki, aci + ki, acj, qout_m, qout_n, q_total,
                                              eici, hash_val[s], tau, in_m, hash_in_val[s],
                                              tot_out, n_m, n_n, nc_v, N_total, d_m, d_n, d_v);
            } else {
                g = move_gain<ACTIVE_OBJECTIVE>(hash_val[s], eici, ki, aci, acj,
                                                qout_m, qout_n, q_total, mass);
            }
            if (g > best_gain ||
                (g == best_gain && ((up_down && cand < best_dst) || (!up_down && cand > best_dst)))) {
                best_gain = g;
                best_dst  = cand;
            }
        }

        // Reduce the per-lane bests across the warp, same tie-break.
        for (int off = 16; off > 0; off >>= 1) {
            const weight_t g_o = warp.shfl_down(best_gain, off);
            const vertex_t d_o = warp.shfl_down(best_dst,  off);
            if (g_o > best_gain ||
                (g_o == best_gain && ((up_down && d_o < best_dst) || (!up_down && d_o > best_dst)))) {
                best_gain = g_o;
                best_dst  = d_o;
            }
        }

        if (warp.thread_rank() == 0) {
            vertex_t final_dst;
            if (directed) {
                final_dst = best_dst;                       // no ID gate (see 5cb048d)
            } else if (up_down) {
                final_dst = best_dst < src_community_id ? best_dst : src_community_id;
            } else {
                final_dst = best_dst > src_community_id ? best_dst : src_community_id;
            }
            shared_device_community_ids_new_[v] = final_dst;
        }
        warp.sync();
    }
}

// Host launcher for the warp-per-vertex path. Single kernel, single stream.
void calculate_eicj_and_move_vertex_warp(
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
    vertex_t* hash_key,
    weight_t* hash_val,
    weight_t* hash_in_val,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    cudaStream_t stream,
    weight_t* s_out_local,
    double tau,
    vertex_t* in_offset,
    edge_t* in_edge,
    weight_t* in_edge_weight,
    weight_t* node_count,
    weight_t* community_node_count,
    weight_t* dangling_flow,
    weight_t* community_dangling_flow,
    weight_t  N_total)
{
    const int block_num = 256;                       // 8 warps per block
    int grid_num = (int) iDivUp(local_vertices, (vertex_t)(block_num / 32));
    if (grid_num < 1) grid_num = 1;
    if (grid_num > 65535) grid_num = 65535;          // grid-stride loop covers the rest

    move_vertices_warp<<<grid_num, block_num, 0, stream>>>(
        begin_vertex_id, local_vertices,
        private_device_offset, private_device_edge, private_device_edge_weight,
        private_device_vertex_weight, private_device_part_vertex_offset,
        shared_device_community_ids_new, shared_device_community_ids,
        shared_device_community_weight, shared_device_community_q_out, Q_sum,
        hash_key, hash_val, hash_in_val,
        mass, my_pe, n_pes, up_down,
        s_out_local, tau, in_offset, in_edge, in_edge_weight,
        node_count, community_node_count, dangling_flow, community_dangling_flow, N_total);
    CUDA_RT_CALL(cudaGetLastError());                // the binned path never checked this
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
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
    double tau = 0.0,
    // level-0 Sum_v F(p_vis_v); nullptr = derive from this level's vertex weights
    weight_t* s_node_level0 = nullptr,
    // directed: per-module node count n_i (original nodes) and the total N, for the
    // unrecorded-teleportation q_out
    weight_t* community_node_count = nullptr,
    weight_t* community_dangling_flow = nullptr,
    weight_t  N_total = 1.0)
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
            local_vertices, tau_f, shared_device_community_weight, shared_device_community_q_out,
            community_node_count, community_dangling_flow, N_total);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
    }

    // 3) assemble codelength -> Q (= -L), Q_sum (= raw Sum_i cut_i)
    void *a3[] = {
            (void *) &mass, (void *) &local_vertices,
            (void *) &shared_device_community_weight, (void *) &shared_device_community_q_out,
            (void *) &private_device_vertex_weight, (void *) &cl_reduce,
            (void *) &my_pe, (void *) &n_pes, (void *) &Q, (void *) &Q_sum,
            (void *) &s_node_level0
    };
    NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_codelength, block_dims, a3, d_shared_mem, &grid_size));
    nvshmem_barrier_all();
    NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_codelength, grid_size, block_dims, a3, d_shared_mem, default_stream));
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}

// ---------------------------------------------------------------------------
// B6: directed coarsening.
//
// Coarsening a directed FLOW graph is only exact if you aggregate the walk flow rather than
// the raw edge weights (proved in tools/directed_coarsen_check.py, error 0.0e+00):
//
//   coarse edge   f(C,D) = sum_{u in C, v in D, u->v} p_u * w(u,v) / s_out(u)
//   coarse flow   p_C    = sum_{v in C} p_v          (== community_weight at phase end)
//   coarse s_out  s_out(C) := p_C
//
// s_out(C) must be p_C and NOT sum_D f(C,D): dangling vertices (out-degree 0 -- wiki-Vote has
// 2187 of them) carry flow but emit none, so the two differ by the dangling mass and using the
// latter costs +2.39 bits. With s_out(C) = p_C the kernels' p_C * f(C,D)/s_out(C) collapses to
// exactly f(C,D), which is the module's true walk exit flow at the ORIGINAL level.
//
// Rescaling the out-CSR to the walk flow is IDEMPOTENT at coarse levels: there edge_weight is
// already f and s_out == vertex_weight == p_C, so p_C * f / p_C = f. The same kernel therefore
// runs before every coarsening.
__global__ void scale_edges_to_walk_flow(
    vertex_t local_vertices,
    vertex_t* private_device_offset,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,   // p_v
    weight_t* s_out
)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t v = grid.thread_rank(); v < local_vertices; v += grid.num_threads()) {
        const weight_t sv = s_out[v];
        const weight_t pv = private_device_vertex_weight[v];
        for (edge_t e = private_device_offset[v]; e < private_device_offset[v + 1]; ++e) {
            private_device_edge_weight[e] = (sv > 0.)
                ? (weight_t)((double)pv * (double)private_device_edge_weight[e] / (double)sv)
                : (weight_t)0.;
        }
    }
}

// p_C = sum_{v in C} p_v, indexed by the DENSE coarse id. Run right after coarsen_graph, while
// community_ids still maps each old vertex to its dense coarse id (renew_community_id_cuda) and
// vertex_weight still holds the old level's p_v.
__global__ void aggregate_coarse_flow(
    vertex_t prev_local_vertices,
    vertex_t* dense_coarse_id,                // old vertex -> DENSE coarse id
    weight_t* private_device_vertex_weight,   // old p_v
    weight_t* coarse_p
)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t v = grid.thread_rank(); v < prev_local_vertices; v += grid.num_threads()) {
        atomicAdd(coarse_p + dense_coarse_id[v], private_device_vertex_weight[v]);
    }
}

// Level 0: a vertex is dangling iff it has no out-links (s_out == 0); its whole visit rate is
// dangling flow. At coarse levels this is carried by aggregate_coarse_flow instead.
__global__ void init_dangling_flow(
    vertex_t local_vertices,
    weight_t* s_out,
    weight_t* p_vis,
    weight_t* dangling_flow
)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t v = grid.thread_rank(); v < local_vertices; v += grid.num_threads()) {
        dangling_flow[v] = (s_out[v] <= 0.) ? p_vis[v] : (weight_t)0.;
    }
}

__global__ void fill_weight(weight_t* a, vertex_t n, weight_t val)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t i = grid.thread_rank(); i < n; i += grid.num_threads()) a[i] = val;
}

// Rebuild the coarse in-CSR by transposing the coarse out-CSR. Done on the host: the coarse
// graph is small (it only shrinks) and this runs once per phase, so it is not worth a device
// sort. -np 1 only, which is also the only regime where the directed in-flow gain is exact.
static void rebuild_coarse_in_csr(
    vertex_t local_vertices,
    edge_t   local_edges,
    vertex_t* d_offset, edge_t* d_edge, weight_t* d_edge_weight,
    vertex_t* d_in_offset, edge_t* d_in_edge, weight_t* d_in_edge_weight,
    cudaStream_t stream)
{
    std::vector<vertex_t> off(local_vertices + 1);
    std::vector<edge_t>   edg(local_edges);
    std::vector<weight_t> wgt(local_edges);
    CUDA_RT_CALL(cudaMemcpy(off.data(), d_offset, sizeof(vertex_t) * (local_vertices + 1), cudaMemcpyDeviceToHost));
    if (local_edges > 0) {
        CUDA_RT_CALL(cudaMemcpy(edg.data(), d_edge, sizeof(edge_t) * local_edges, cudaMemcpyDeviceToHost));
        CUDA_RT_CALL(cudaMemcpy(wgt.data(), d_edge_weight, sizeof(weight_t) * local_edges, cudaMemcpyDeviceToHost));
    }

    std::vector<vertex_t> in_off(local_vertices + 1, 0);
    for (edge_t e = 0; e < local_edges; ++e) in_off[edg[e] + 1]++;
    for (vertex_t v = 0; v < local_vertices; ++v) in_off[v + 1] += in_off[v];

    std::vector<edge_t>   in_edg(local_edges);
    std::vector<weight_t> in_wgt(local_edges);
    std::vector<vertex_t> cursor(in_off.begin(), in_off.end() - 1);
    for (vertex_t u = 0; u < local_vertices; ++u) {
        for (edge_t e = off[u]; e < off[u + 1]; ++e) {
            const vertex_t dst = edg[e];
            const vertex_t pos = cursor[dst]++;
            in_edg[pos] = u;              // in_edge holds the SOURCE id (global == local at -np 1)
            in_wgt[pos] = wgt[e];
        }
    }

    CUDA_RT_CALL(cudaMemcpy(d_in_offset, in_off.data(), sizeof(vertex_t) * (local_vertices + 1), cudaMemcpyHostToDevice));
    if (local_edges > 0) {
        CUDA_RT_CALL(cudaMemcpy(d_in_edge, in_edg.data(), sizeof(edge_t) * local_edges, cudaMemcpyHostToDevice));
        CUDA_RT_CALL(cudaMemcpy(d_in_edge_weight, in_wgt.data(), sizeof(weight_t) * local_edges, cudaMemcpyHostToDevice));
    }
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
}
}  // namespace louvain

void louvain::run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter,
                           const int max_phases, const double tau, const std::string &out_path,
                           const std::string &move_mode) {
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();
    const bool use_warp_move = (move_mode != "binned");

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

    // Sum_v F(p_vis_v) over the ORIGINAL vertices, captured at Phase 0 and reused by every
    // later phase so that L stays the codelength of the partition (see compute_codelength).
    weight_t *s_node_level0 = nullptr;
    double prev_phase_final_L = 0.0;

    // B6: p_C for the level produced by the last coarsening. Non-null from Phase 1 on in a
    // directed run; it overrides both vertex_weight (node flow) and s_out at the coarse level.
    weight_t *coarse_p = nullptr;

    // Flat map from an ORIGINAL vertex to its community at the current level. Each phase
    // composes this with the level's assignment, so after the last phase it is the full
    // hierarchical partition. Only meaningful at -np 1 (community_ids is distributed).
    const vertex_t original_vertices = total_vertices;
    std::vector<vertex_t> orig_to_comm;
    if (!out_path.empty()) {
        if (n_pes != 1) {
            if (my_pe == 0)
                printf("[WARN] -out is only supported at -np 1 (community_ids is distributed); skipping.\n");
        } else {
            orig_to_comm.resize(original_vertices);
            std::iota(orig_to_comm.begin(), orig_to_comm.end(), 0);
        }
    }

    // private memory
    auto *private_device_offset = gpuGraph->get_private_device_offset_();
    auto *private_device_edge = gpuGraph->get_private_device_edge_();
    auto *private_device_edge_weight = gpuGraph->get_private_device_edge_weight_();
    auto *private_device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    auto *private_device_vertex_weight = gpuGraph->get_private_device_vertex_weight_();

    // init bins
    BIN* bins = new BIN(BIN_NUM, local_vertices);

    // Global per-vertex hash scratch for the warp-per-vertex move kernel: vertex v owns the
    // slice [2*offset[v], 2*offset[v+1]), i.e. capacity 2*deg(v). Sized from the ORIGINAL
    // edge count, which upper-bounds every coarse level (coarsening only merges edges).
    vertex_t *mv_hash_key    = nullptr;
    weight_t *mv_hash_val    = nullptr;
    weight_t *mv_hash_in_val = nullptr;
    if (use_warp_move) {
        const size_t hash_slots = (size_t) local_edges * 2;
        CUDA_RT_CALL(cudaMalloc((void **) &mv_hash_key,    sizeof(vertex_t) * hash_slots));
        CUDA_RT_CALL(cudaMalloc((void **) &mv_hash_val,    sizeof(weight_t) * hash_slots));
        CUDA_RT_CALL(cudaMalloc((void **) &mv_hash_in_val, sizeof(weight_t) * hash_slots));
        if (my_pe == 0) {
            printf("local-moving kernel: warp-per-vertex (hash scratch %.1f MB)\n",
                   (double)(hash_slots * (sizeof(vertex_t) + 2 * sizeof(weight_t))) / (1024.0 * 1024.0));
        }
    } else if (my_pe == 0) {
        printf("local-moving kernel: degree-binned tile/block (legacy)\n");
    }

    // directed-mode PageRank scratch
    const bool directed = gpuGraph->is_directed_();

    if (directed && n_pes > 1 && my_pe == 0) {
        printf("[WARN] directed multi-level (B6) requires -np 1: the coarse in-CSR rebuild and the\n"
               "       in-flow gain both need a local view. At -np %d only Phase 0 is directed;\n"
               "       later phases fall back to the undirected objective.\n", n_pes);
    }

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

    // Unrecorded teleportation needs n_i, the number of ORIGINAL nodes per module. Like the
    // flow, it has to be carried through coarsening (a coarse super-node stands for many
    // original nodes), and like community_weight it has to be maintained incrementally across
    // moves and rolled back on a rejected pass.
    const weight_t N_total = (weight_t) max_total_vertices;
    weight_t *node_count           = nullptr;   // per-vertex nc[v] (1 at level 0)
    weight_t *community_node_count = nullptr;   // per-module n_i  (NVSHMEM: candidates may be remote)
    weight_t *cn_backup            = nullptr;   // n_i of the last accepted partition
    weight_t *coarse_nc            = nullptr;   // nc of the level produced by the last coarsening
    // Dangling flow: a node with no out-links must teleport (probability 1, not tau), so its
    // exit flow is charged in full. Tracked exactly like the node count.
    weight_t *dangling_flow           = nullptr;
    weight_t *community_dangling_flow = nullptr;
    weight_t *cd_backup               = nullptr;
    weight_t *coarse_df               = nullptr;
    if (directed) {
        CUDA_RT_CALL(cudaMalloc((void **) &node_count, max_total_vertices * sizeof(weight_t)));
        community_node_count = (weight_t *) nvshmem_malloc(max_total_vertices * sizeof(weight_t));
        CUDA_RT_CALL(cudaMalloc((void **) &cn_backup, max_total_vertices * sizeof(weight_t)));
        CUDA_RT_CALL(cudaMalloc((void **) &dangling_flow, max_total_vertices * sizeof(weight_t)));
        community_dangling_flow = (weight_t *) nvshmem_malloc(max_total_vertices * sizeof(weight_t));
        CUDA_RT_CALL(cudaMalloc((void **) &cd_backup, max_total_vertices * sizeof(weight_t)));
    }

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

        // B6: directed coarsening keeps the flow model at every level (walk-flow edges,
        // p_C node flow, s_out(C) = p_C, rebuilt in-CSR), so directed is no longer a
        // Phase-0-only mode. It requires -np 1, which is also the only regime where the
        // directed in-flow gain is exact (remote in-neighbours have no symmetric p[u]).
        bool phase_directed = directed && (n_pes == 1 || phase_num == 0);

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

        // 2a. Directed: node counts for the unrecorded-teleportation term. 1 per vertex at
        //     level 0; at a coarse level, nc[C] = |C| in ORIGINAL nodes (carried by coarsening).
        if (directed) {
            if (phase_num == 0 || coarse_nc == nullptr) {
                fill_weight<<<80, 1024, 0, default_stream>>>(node_count, local_vertices, (weight_t)1.0);
            } else {
                copy<weight_t><<<80, 1024, 0, default_stream>>>(coarse_nc, node_count, local_vertices);
            }
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
        }

        // 2b. Directed mode: replace k_v with PageRank ergodic visit probability p_vis[v].
        //     Normalize k_v to probability units first (uniform init), then iterate.
        if (phase_directed && phase_num > 0 && coarse_p != nullptr) {
            // B6 coarse level: the flow is already known exactly -- p_C was accumulated at the
            // last coarsening. Do NOT re-run PageRank (teleportation over N_coarse nodes is a
            // different Markov chain, so it would not reproduce p_C = sum_{v in C} p_v), and do
            // NOT keep reduce_vertices_weights' answer (that is sum_D f(C,D), which is short by
            // the dangling mass). s_out(C) := p_C makes the kernels' p_C*f/s_out collapse to f.
            copy<weight_t><<<80, 1024, 0, default_stream>>>(coarse_p, private_device_vertex_weight, local_vertices);
            copy<weight_t><<<80, 1024, 0, default_stream>>>(coarse_p, pr_s_out, local_vertices);
            copy<weight_t><<<80, 1024, 0, default_stream>>>(coarse_df, dangling_flow, local_vertices);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            mass = 1.0;
        } else if (phase_directed) {
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
            // After convergence: p_vis is in private_device_vertex_weight (normalized, Σ = 1).
            // A vertex with no out-links is dangling: all of its flow leaves by teleportation.
            init_dangling_flow<<<80, 1024, 0, default_stream>>>(
                local_vertices, pr_s_out, private_device_vertex_weight, dangling_flow);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
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
        if (directed) {
            // same seeding for n_i and d_i: singleton partition -> per-module == per-vertex
            copy<weight_t><<<80, 1024, 0, default_stream>>>(node_count, community_node_count, local_vertices);
            copy<weight_t><<<80, 1024, 0, default_stream>>>(dangling_flow, community_dangling_flow, local_vertices);
        }
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        // 3. Compute the initial codelength score (-L) of this phase
        launch_compute_codelength(mass, local_vertices, total_vertices,
                                  private_device_offset, private_device_edge, private_device_edge_weight,
                                  private_device_part_vertex_offset, shared_device_community_ids,
                                  shared_device_community_weight, shared_device_community_q_out,
                                  shared_device_community_delta_weight, private_device_vertex_weight,
                                  cl_reduce, Q, Q_sum, my_pe, n_pes, block_dims, d_shared_mem, default_stream,
                                  phase_directed ? pr_s_out : nullptr, phase_directed ? tau : 0.0,
                                  s_node_level0, community_node_count, community_dangling_flow, N_total);

        CUDA_RT_CALL(cudaMemcpy(&new_Q, Q, sizeof(weight_t) , cudaMemcpyDeviceToHost));

        // Capture Sum_v F(p_vis_v) of the ORIGINAL vertices once, and reuse it for every
        // later phase. Coarsening merges nodes into super-nodes with larger visit rates, so
        // re-deriving this term per level silently changes the objective: L stops being the
        // codelength of the partition and the outer phase loop compares incomparable numbers.
        // With B6 the directed levels keep the same flow objective (mass = 1 throughout), so
        // the same constant is valid there too -- but only where B6 runs. A directed run at
        // -np > 1 still drops to the undirected objective (different units) after Phase 0, so
        // no single constant is valid there.
        const bool objective_is_stable = (!directed || n_pes == 1);
        if (phase_num == 0 && objective_is_stable && s_node_level0 == nullptr) {
            CUDA_RT_CALL(cudaMalloc((void **) &s_node_level0, sizeof(weight_t)));
            CUDA_RT_CALL(cudaMemcpy(s_node_level0, cl_reduce + 3, sizeof(weight_t),
                                    cudaMemcpyDeviceToDevice));
        }

        // Phase continuity: a singleton partition of the coarse graph IS the partition the
        // previous phase ended on, so its codelength must be identical. If this trips, the
        // objective is being re-based across levels and the multi-level L is meaningless.
        if (my_pe == 0 && phase_num > 0 && objective_is_stable) {
            const double L_now = -(double)new_Q;
            if (fabs(L_now - prev_phase_final_L) > 1.0e-6) {
                printf("[WARN] codelength not conserved across coarsening: phase %d ended at "
                       "L=%.6f but phase %d starts at L=%.6f (delta %+.6f)\n",
                       phase_num - 1, prev_phase_final_L, phase_num, L_now, L_now - prev_phase_final_L);
            }
        }

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
        if (directed) {
            CUDA_RT_CALL(cudaMemcpyAsync(cn_backup, community_node_count,
                                         local_vertices * sizeof(weight_t),
                                         cudaMemcpyDeviceToDevice, default_stream));
            CUDA_RT_CALL(cudaMemcpyAsync(cd_backup, community_dangling_flow,
                                         local_vertices * sizeof(weight_t),
                                         cudaMemcpyDeviceToDevice, default_stream));
        }
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
            if (use_warp_move) {
                calculate_eicj_and_move_vertex_warp(local_vertices,
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
                                                    mv_hash_key,
                                                    mv_hash_val,
                                                    mv_hash_in_val,
                                                    mass,
                                                    my_pe,
                                                    n_pes,
                                                    up_down,
                                                    default_stream,
                                                    phase_directed ? pr_s_out : nullptr,
                                                    phase_directed ? tau : 0.0,
                                                    phase_directed ? pr_in_offset : nullptr,
                                                    phase_directed ? pr_in_edge : nullptr,
                                                    phase_directed ? pr_in_edge_weight : nullptr,
                                                    node_count,
                                                    community_node_count,
                                                    dangling_flow,
                                                    community_dangling_flow,
                                                    N_total);
            } else {
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
            }



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

            // b2) directed: the same incremental update for the module NODE COUNTS n_i, which
            // the unrecorded-teleportation q_out depends on. Identical kernels, different
            // arrays -- the delta buffer is re-zeroed by the kernel, so it can be reused.
            if (directed) {
                void *kernel_args_ccn[] = {
                        (void *) &local_vertices,
                        (void *) &total_vertices,
                        (void *) &node_count,
                        (void *) &shared_device_community_ids,
                        (void *) &shared_device_community_ids_new,
                        (void *) &community_node_count,
                        (void *) &shared_device_community_delta_weight,
                        (void *) &private_device_part_vertex_offset,
                        (void *) &my_pe,
                        (void *) &n_pes
                };
                NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight_local_atomic, block_dims, kernel_args_ccn, d_shared_mem, &grid_size));
                nvshmem_barrier_all();
                NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight_local_atomic, grid_size, block_dims, kernel_args_ccn, d_shared_mem, default_stream));
                nvshmemx_barrier_all_on_stream(default_stream);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

                NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight, block_dims, kernel_args_ccn, d_shared_mem, &grid_size));
                nvshmem_barrier_all();
                NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight, grid_size, block_dims, kernel_args_ccn, d_shared_mem, default_stream));
                nvshmemx_barrier_all_on_stream(default_stream);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

                // ... and the same again for the per-module DANGLING flow d_i.
                void *kernel_args_ccd[] = {
                        (void *) &local_vertices,
                        (void *) &total_vertices,
                        (void *) &dangling_flow,
                        (void *) &shared_device_community_ids,
                        (void *) &shared_device_community_ids_new,
                        (void *) &community_dangling_flow,
                        (void *) &shared_device_community_delta_weight,
                        (void *) &private_device_part_vertex_offset,
                        (void *) &my_pe,
                        (void *) &n_pes
                };
                NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight_local_atomic, block_dims, kernel_args_ccd, d_shared_mem, &grid_size));
                nvshmem_barrier_all();
                NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight_local_atomic, grid_size, block_dims, kernel_args_ccd, d_shared_mem, default_stream));
                nvshmemx_barrier_all_on_stream(default_stream);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

                NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight, block_dims, kernel_args_ccd, d_shared_mem, &grid_size));
                nvshmem_barrier_all();
                NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight, grid_size, block_dims, kernel_args_ccd, d_shared_mem, default_stream));
                nvshmemx_barrier_all_on_stream(default_stream);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            }

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
                                      phase_directed ? pr_s_out : nullptr, phase_directed ? tau : 0.0,
                                      s_node_level0, community_node_count, community_dangling_flow, N_total);

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
                if (directed) {
                    CUDA_RT_CALL(cudaMemcpyAsync(cn_backup, community_node_count,
                                                 local_vertices * sizeof(weight_t),
                                                 cudaMemcpyDeviceToDevice, default_stream));
                    CUDA_RT_CALL(cudaMemcpyAsync(cd_backup, community_dangling_flow,
                                                 local_vertices * sizeof(weight_t),
                                                 cudaMemcpyDeviceToDevice, default_stream));
                }
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
                if (directed) {
                    CUDA_RT_CALL(cudaMemcpyAsync(community_node_count, cn_backup,
                                                 local_vertices * sizeof(weight_t),
                                                 cudaMemcpyDeviceToDevice, default_stream));
                    CUDA_RT_CALL(cudaMemcpyAsync(community_dangling_flow, cd_backup,
                                                 local_vertices * sizeof(weight_t),
                                                 cudaMemcpyDeviceToDevice, default_stream));
                }
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

        prev_phase_final_L = -(double)new_Q;

        // Compose this level's assignment into the flat original-vertex -> community map.
        // community_ids holds ids in this level's vertex space and is not dense; coarsening
        // compacts it (renew_community_id_cuda: dense = scan(used)[old] - 1). Reproduce that
        // same compaction here so orig_to_comm stays in the NEXT level's vertex space.
        if (!orig_to_comm.empty()) {
            std::vector<vertex_t> level_comm(local_vertices);
            CUDA_RT_CALL(cudaMemcpy(level_comm.data(), shared_device_community_ids,
                                    sizeof(vertex_t) * local_vertices, cudaMemcpyDeviceToHost));
            std::vector<vertex_t> dense(local_vertices, 0);
            for (vertex_t v = 0; v < local_vertices; ++v) dense[level_comm[v]] = 1;
            vertex_t run_sum = 0;
            for (vertex_t c = 0; c < local_vertices; ++c) { run_sum += dense[c]; dense[c] = run_sum - 1; }
            for (vertex_t v = 0; v < original_vertices; ++v)
                orig_to_comm[v] = dense[level_comm[orig_to_comm[v]]];
        }

        // Phase continuation is driven by this phase's own improvement (final vs. initial
        // score). With the level-0 node-entropy term pinned (s_node_level0) the codelength is
        // conserved across coarsening, so comparing across levels would work too.
        Q_old_host = phase_initial_score;
        Q_host = new_Q;
        if ((Q_host - Q_old_host) <= threshold) break;

        start_phase = MPI_Wtime();

        // B6: coarsen the FLOW, not the raw weights. Rescale the out-CSR to the walk flow
        // p_u*w(u,v)/s_out(u) so the existing aggregation produces f(C,D) directly. Idempotent
        // at coarse levels (edge_weight is already f and s_out == p_C there).
        const vertex_t prev_local_vertices = local_vertices;
        if (phase_directed) {
            scale_edges_to_walk_flow<<<80, 1024, 0, default_stream>>>(
                local_vertices, private_device_offset, private_device_edge_weight,
                private_device_vertex_weight, pr_s_out);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
        }

        coarsen_graph_mg::coarsen_graph(hostGraph, gpuGraph, my_pe, n_pes, default_stream, streams, symbolic_time, numeric_time);

        stop_phase = MPI_Wtime();
        coarsen_graph_total_time += (stop_phase - start_phase);
        phase_num++;
        if (my_pe == 0) {
            printf("Coarsen graph execution time: %f s, %f ms\n", (stop_phase - start_phase), double((stop_phase - start_phase) * 1000));
        }

        local_vertices = gpuGraph->get_local_vertices_();
        total_vertices = gpuGraph->get_total_vertices_();

        // B6: finish the coarse flow graph. coarsen_graph has just aggregated the (rescaled)
        // edge weights into f(C,D), and community_ids still maps each OLD vertex to its dense
        // coarse id (renew_community_id_cuda) while vertex_weight still holds the old p_v --
        // so this is the one window in which p_C can be accumulated.
        if (phase_directed) {
            CUDA_RT_CALL(cudaFree(coarse_p));
            CUDA_RT_CALL(cudaMalloc((void **) &coarse_p, sizeof(weight_t) * local_vertices));
            fill_weight<<<80, 1024, 0, default_stream>>>(coarse_p, local_vertices, (weight_t)0.);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            // NB: coarsen_graph renumbers into the _NEW buffer, not community_ids --
            // shared_device_community_ids_global == get_shared_device_community_ids_new_().
            // So the dense old-vertex -> coarse-id map lives there.
            aggregate_coarse_flow<<<80, 1024, 0, default_stream>>>(
                prev_local_vertices, shared_device_community_ids_new,
                private_device_vertex_weight, coarse_p);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            // n_i counts ORIGINAL nodes, so the node count is carried through coarsening the
            // same way as the flow: nc[C] = sum_{v in C} nc[v]. Without this the teleport
            // factor (N - n_i)/N would be computed over super-nodes and the objective would be
            // silently re-based at every level, exactly like the node-entropy term was.
            CUDA_RT_CALL(cudaFree(coarse_nc));
            CUDA_RT_CALL(cudaMalloc((void **) &coarse_nc, sizeof(weight_t) * local_vertices));
            fill_weight<<<80, 1024, 0, default_stream>>>(coarse_nc, local_vertices, (weight_t)0.);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            aggregate_coarse_flow<<<80, 1024, 0, default_stream>>>(
                prev_local_vertices, shared_device_community_ids_new,
                node_count, coarse_nc);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            // ... and the dangling flow, for the same reason: d_i must count ORIGINAL dangling
            // nodes, so a super-node has to remember how much of its flow is dangling.
            CUDA_RT_CALL(cudaFree(coarse_df));
            CUDA_RT_CALL(cudaMalloc((void **) &coarse_df, sizeof(weight_t) * local_vertices));
            fill_weight<<<80, 1024, 0, default_stream>>>(coarse_df, local_vertices, (weight_t)0.);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            aggregate_coarse_flow<<<80, 1024, 0, default_stream>>>(
                prev_local_vertices, shared_device_community_ids_new,
                dangling_flow, coarse_df);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            // The coarse graph is directed, so the in-CSR must be rebuilt (coarsen only emits
            // the out-CSR). Coarse edges <= original edges, so the level-0 in-CSR buffers fit.
            rebuild_coarse_in_csr(local_vertices, gpuGraph->get_local_edges_(),
                                  private_device_offset, private_device_edge, private_device_edge_weight,
                                  pr_in_offset, pr_in_edge, pr_in_edge_weight, default_stream);
        }

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
    if (!orig_to_comm.empty() && my_pe == 0) {
        std::ofstream out(out_path);
        if (!out) {
            printf("[WARN] could not open -out file '%s'\n", out_path.c_str());
        } else {
            for (vertex_t v = 0; v < original_vertices; ++v)
                out << v << ' ' << orig_to_comm[v] << '\n';
            out.close();
            vertex_t n_comm = 0;
            {
                std::vector<char> seen(original_vertices, 0);
                for (vertex_t v = 0; v < original_vertices; ++v) {
                    if (!seen[orig_to_comm[v]]) { seen[orig_to_comm[v]] = 1; ++n_comm; }
                }
            }
            printf("Wrote partition to %s  (%u vertices, %u communities)\n",
                   out_path.c_str(), original_vertices, n_comm);
        }
    }

    CUDA_RT_CALL(cudaFree(cw_backup));
    CUDA_RT_CALL(cudaFree(cq_backup));
    CUDA_RT_CALL(cudaFree(qsum_backup));
    if (s_node_level0 != nullptr) CUDA_RT_CALL(cudaFree(s_node_level0));
    if (coarse_p != nullptr) CUDA_RT_CALL(cudaFree(coarse_p));
    if (coarse_nc != nullptr) CUDA_RT_CALL(cudaFree(coarse_nc));
    if (node_count != nullptr) CUDA_RT_CALL(cudaFree(node_count));
    if (cn_backup != nullptr) CUDA_RT_CALL(cudaFree(cn_backup));
    if (community_node_count != nullptr) nvshmem_free(community_node_count);
    if (coarse_df != nullptr) CUDA_RT_CALL(cudaFree(coarse_df));
    if (dangling_flow != nullptr) CUDA_RT_CALL(cudaFree(dangling_flow));
    if (cd_backup != nullptr) CUDA_RT_CALL(cudaFree(cd_backup));
    if (community_dangling_flow != nullptr) nvshmem_free(community_dangling_flow);
    if (mv_hash_key    != nullptr) CUDA_RT_CALL(cudaFree(mv_hash_key));
    if (mv_hash_val    != nullptr) CUDA_RT_CALL(cudaFree(mv_hash_val));
    if (mv_hash_in_val != nullptr) CUDA_RT_CALL(cudaFree(mv_hash_in_val));
    nvshmem_free(Q);
    nvshmem_free(Q_sum);
    nvshmem_free(cl_reduce);
    if (directed) {
        CUDA_RT_CALL(cudaFree(pr_p_new));
        nvshmem_free(pr_reduce);
        nvshmem_free(pr_p_cur);
    }
}

