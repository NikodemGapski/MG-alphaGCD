#ifndef HOST_GRAPH_CUH
#define HOST_GRAPH_CUH

#include "../common.h"
#include "../mmio/mmio_wrapper.h"
#include <stdlib.h>

class HostGraph
{
private:
     vertex_t total_vertices_;
     edge_t total_edge_;
     weight_t mass_;

     vertex_t *host_offset_;
     edge_t *host_edge_;
     weight_t *host_edge_weight_;

     // Directed-mode extras (populated only when directed_ == true)
     vertex_t *host_in_offset_;
     edge_t   *host_in_edge_;
     weight_t *host_in_edge_weight_;
     weight_t *host_s_out_;           // s_out[v] = sum of outgoing edge weights
     edge_t    total_in_edge_;

     bool directed_;  // true → keep directed edges and build in-CSR (Part B1)

public:
    // directed=false → legacy symmetric path (default, backward-compatible)
    HostGraph(char *graph_path, int my_pe, bool directed = false);
    HostGraph(int random_vertex_num, double sparsity, int my_pe);
    ~HostGraph(){
        free(host_offset_);
        free(host_edge_);
        free(host_edge_weight_);
        if (directed_) {
            delete[] host_in_offset_;
            delete[] host_in_edge_;
            delete[] host_in_edge_weight_;
            delete[] host_s_out_;
        }
    }
    vertex_t get_total_vertices_();
    edge_t get_total_edge_();
    void set_total_vertices_(vertex_t total_vertices_);
    void set_total_edge_(vertex_t total_edge_);
    weight_t get_mass_();
    bool is_directed_() const { return directed_; }

    vertex_t* get_host_offset_();
    edge_t* get_host_edge_();
    weight_t* get_host_edge_weight_();

    // Directed-mode accessors (only valid when is_directed_() == true)
    vertex_t* get_host_in_offset_()   const { return host_in_offset_; }
    edge_t*   get_host_in_edge_()     const { return host_in_edge_; }
    weight_t* get_host_in_weight_()   const { return host_in_edge_weight_; }
    weight_t* get_host_s_out_()       const { return host_s_out_; }
    edge_t    get_total_in_edge_()    const { return total_in_edge_; }

    void load_graph_mtx(char *graph_path);
    void randomly_generate_graph(int random_vertex_num, double sparsity);
    void compute_total_edge_weight();
};

#endif //HOST_GRAPH_CUH
