#ifndef PRIVATE_GPU_GRAPH_CUH
#define PRIVATE_GPU_GRAPH_CUH
#include "../common.h"
#include "./host_graph.h"
class GpuGraph{
private:
    vertex_t total_vertices_;
    edge_t total_edges_;
    vertex_t local_vertices_;
    edge_t local_edges_;
    vertex_t* part_vertex_offset_; /* Vertex partitioning result among GPUs */
    edge_t len_edges_array_; /* The actual length of the `private_device_edge_` array -- `avg_edges + total_vertices_` */
    weight_t mass_;

    // private — out-CSR (both NVSHMEM so any PE can read remote vertices)
    vertex_t* private_device_offset_;
    edge_t*   private_device_edge_;
    weight_t* private_device_edge_weight_;
    vertex_t* private_device_part_vertex_offset_;
    weight_t* private_device_vertex_weight_;   // p_vis[v] (directed) or k_v (undirected)

    // in-CSR (transpose adjacency, only allocated in directed mode)
    vertex_t* private_device_in_offset_;
    edge_t*   private_device_in_edge_;
    weight_t* private_device_in_edge_weight_;
    weight_t* private_device_s_out_;           // out-strength per vertex
    edge_t    len_in_edges_array_;
    bool      directed_;

    // shared
    weight_t* shared_device_community_weight_;
    weight_t* shared_device_community_delta_weight_;
    weight_t* shared_device_community_q_out_;        /* Map equation: per-module exit (cut) weight, raw units */
    vertex_t* shared_device_community_ids_;
    vertex_t* shared_device_community_ids_new_;


public:
    GpuGraph(int n_pes, HostGraph *hostGraph);
    ~GpuGraph(){
        nvshmem_free(private_device_offset_);
        CUDA_RT_CALL(cudaFree(private_device_edge_));
        CUDA_RT_CALL(cudaFree(private_device_edge_weight_));
        CUDA_RT_CALL(cudaFree(private_device_part_vertex_offset_));
        CUDA_RT_CALL(cudaFree(private_device_vertex_weight_));

        if (directed_) {
            nvshmem_free(private_device_in_offset_);
            nvshmem_free(private_device_in_edge_);
            nvshmem_free(private_device_in_edge_weight_);
            nvshmem_free(private_device_s_out_);
        }

        nvshmem_free(shared_device_community_weight_);
        nvshmem_free(shared_device_community_delta_weight_);
        nvshmem_free(shared_device_community_q_out_);
        nvshmem_free(shared_device_community_ids_);
        nvshmem_free(shared_device_community_ids_new_);
    }

    // get array point
    vertex_t* get_part_vertex_offset_();
    vertex_t* get_private_device_part_vertex_offset_();

    vertex_t* get_private_device_offset_();
    edge_t* get_private_device_edge_();
    weight_t* get_private_device_edge_weight_();

    // in-CSR accessors (directed mode only)
    vertex_t* get_private_device_in_offset_()      { return private_device_in_offset_; }
    edge_t*   get_private_device_in_edge_()        { return private_device_in_edge_; }
    weight_t* get_private_device_in_edge_weight_() { return private_device_in_edge_weight_; }
    weight_t* get_private_device_s_out_()          { return private_device_s_out_; }
    bool      is_directed_()                       { return directed_; }

    void set_private_device_in_edge_(edge_t *p)        { private_device_in_edge_ = p; }
    void set_private_device_in_edge_weight_(weight_t *p){ private_device_in_edge_weight_ = p; }
    void set_private_device_s_out_(weight_t *p)        { private_device_s_out_ = p; }
    void set_len_in_edges_array_(edge_t n)             { len_in_edges_array_ = n; }

    edge_t get_len_edges_array_();
    weight_t* get_private_device_vertex_weight_();

    vertex_t* get_shared_device_community_ids_();
    vertex_t* get_shared_device_community_ids_new_();
    weight_t* get_shared_device_community_weight_();
    weight_t* get_shared_device_community_delta_weight_();
    weight_t* get_shared_device_community_q_out_();

    vertex_t get_total_vertices_();
    edge_t get_total_edges_();
    vertex_t get_local_vertices_();
    edge_t get_local_edges_();
    weight_t get_mass_();

    void set_local_vertices_(vertex_t local_vertices);
    void set_local_edges_(edge_t local_edges);
    void set_total_vertices_(vertex_t total_vertices_);
    void set_total_edges_(vertex_t total_edges_);
    void set_len_edges_array_(edge_t len_edges_array);
    void set_private_device_edge_(edge_t* private_device_edge);
    void set_private_device_edge_weight_(weight_t* private_device_edge_weight);
};

#endif //PRIVATE_GPU_GRAPH_CUH
