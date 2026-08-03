#include "../../include/graph/host_graph.h"

#include <sys/stat.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

// --------------------------------------------------------------------------- //
// Binary CSR cache (undirected path only)
//
// Parsing a multi-GB .mtx costs minutes: the parse is a single-threaded
// fgets/atoi loop, and the symmetric branch then does a stable sort over 2*nnz.
// A benchmark campaign re-runs the same graph repeatedly and pays that every
// time for nothing. After the first successful load we dump the CSR beside the
// .mtx and reuse it whenever it is at least as new as the .mtx.
//
// Weights are deliberately NOT stored: both loader branches hardcode every
// weight to 1.0, so they are regenerated on read. That saves 8 B/arc -- 8 GB on
// a billion-arc graph.
//
// Set MG_GCD_NO_CSR_CACHE=1 to bypass entirely (used to prove the cached and
// uncached paths produce identical partitions).
// --------------------------------------------------------------------------- //
namespace {

const uint64_t kCacheMagic   = 0x4D47435352763031ULL;   // "MGCSRv01"
const uint64_t kCacheVersion = 1;                       // bump if loader semantics change

std::string csr_cache_path(const char *graph_path) {
    return std::string(graph_path) + ".csr.bin";
}

bool cache_disabled() {
    const char *e = getenv("MG_GCD_NO_CSR_CACHE");
    return e && *e && *e != '0';
}

// Only trust a cache that is at least as new as the graph it came from.
bool cache_is_fresh(const std::string &cache, const char *graph_path) {
    struct stat cs{}, gs{};
    if (stat(cache.c_str(), &cs) != 0) return false;
    if (stat(graph_path, &gs) != 0) return false;
    return cs.st_mtime >= gs.st_mtime;
}

bool try_load_csr_cache(const char *graph_path, vertex_t *n_out, edge_t *e_out,
                        vertex_t **offset_out, edge_t **col_out, weight_t **val_out) {
    if (cache_disabled()) return false;
    const std::string cache = csr_cache_path(graph_path);
    if (!cache_is_fresh(cache, graph_path)) return false;

    FILE *f = fopen(cache.c_str(), "rb");
    if (!f) return false;

    uint64_t magic = 0, version = 0, n64 = 0, e64 = 0;
    bool ok = fread(&magic, 8, 1, f) == 1 && fread(&version, 8, 1, f) == 1 &&
              fread(&n64, 8, 1, f) == 1 && fread(&e64, 8, 1, f) == 1 &&
              magic == kCacheMagic && version == kCacheVersion;
    // vertex_t / edge_t are uint32; refuse a cache that would not fit them
    if (ok && (n64 > 0xFFFFFFFFULL || e64 > 0xFFFFFFFFULL)) ok = false;
    if (!ok) { fclose(f); return false; }

    const size_t n = (size_t) n64, e = (size_t) e64;
    // malloc, not new[]: ~HostGraph frees these with free()
    vertex_t *offset = (vertex_t *) malloc(sizeof(vertex_t) * (n + 1));
    edge_t   *col    = (edge_t *)   malloc(sizeof(edge_t) * e);
    weight_t *val    = (weight_t *) malloc(sizeof(weight_t) * e);
    if (!offset || !col || !val) {
        free(offset); free(col); free(val); fclose(f);
        return false;
    }

    ok = fread(offset, sizeof(vertex_t), n + 1, f) == n + 1 &&
         fread(col, sizeof(edge_t), e, f) == e;
    fclose(f);
    if (!ok) { free(offset); free(col); free(val); return false; }

    for (size_t i = 0; i < e; ++i) val[i] = 1.0;

    *n_out = (vertex_t) n64; *e_out = (edge_t) e64;
    *offset_out = offset; *col_out = col; *val_out = val;
    return true;
}

// Best effort: a failure here costs nothing but a slow load next time.
// Written to a temp file and renamed so an interrupted write can never leave a
// truncated cache that a later run would trust.
void write_csr_cache(const char *graph_path, vertex_t n, edge_t e,
                     const vertex_t *offset, const edge_t *col) {
    if (cache_disabled()) return;
    const std::string cache = csr_cache_path(graph_path);
    const std::string tmp = cache + ".tmp";

    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    uint64_t magic = kCacheMagic, version = kCacheVersion, n64 = n, e64 = e;
    bool ok = fwrite(&magic, 8, 1, f) == 1 && fwrite(&version, 8, 1, f) == 1 &&
              fwrite(&n64, 8, 1, f) == 1 && fwrite(&e64, 8, 1, f) == 1 &&
              fwrite(offset, sizeof(vertex_t), (size_t) n + 1, f) == (size_t) n + 1 &&
              fwrite(col, sizeof(edge_t), (size_t) e, f) == (size_t) e;
    ok = (fclose(f) == 0) && ok;
    if (ok) {
        if (rename(tmp.c_str(), cache.c_str()) != 0) remove(tmp.c_str());
    } else {
        remove(tmp.c_str());
    }
}

}  // namespace

HostGraph::HostGraph(char *graph_path, int my_pe, bool directed):
total_vertices_(0), total_edge_(0), host_offset_(nullptr),
host_edge_(nullptr), host_edge_weight_(nullptr), mass_(0),
host_in_offset_(nullptr), host_in_edge_(nullptr), host_in_edge_weight_(nullptr),
host_s_out_(nullptr), total_in_edge_(0), directed_(directed)
{
    if (directed_) {
        // Use the directed loader: keeps edges as-is, builds in-CSR and s_out.
        edge_t nnz_in;
        loadMMDirectedSparseMatrix(graph_path,
            &total_vertices_,
            &host_offset_, &host_edge_, &host_edge_weight_, &total_edge_,
            &host_in_offset_, &host_in_edge_, &host_in_edge_weight_, &nnz_in,
            &host_s_out_);
        total_in_edge_ = nnz_in;
    } else {
        load_graph_mtx(graph_path);
    }
    if (my_pe == 0) {
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "Input graph" << std::setw(25) << "Num. vertices (n)" << std::setw(25) << "Num. edges (M)" << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << graph_path << std::setw(25) << total_vertices_ << std::setw(25) << total_edge_ << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
    }
}

HostGraph::HostGraph(int random_vertex_num, double sparsity, int my_pe):
total_vertices_(random_vertex_num), total_edge_(0), host_offset_(nullptr),
host_edge_(nullptr), host_edge_weight_(nullptr), mass_(0),
host_in_offset_(nullptr), host_in_edge_(nullptr), host_in_edge_weight_(nullptr),
host_s_out_(nullptr), total_in_edge_(0), directed_(false)
{
    randomly_generate_graph(random_vertex_num, sparsity);
    if (my_pe == 0) {
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "Input graph" << std::setw(25) << "Num. vertices (n)" << std::setw(25) << "Num. edges (M)" << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "random" << std::setw(25) << total_vertices_ << std::setw(25) << total_edge_ << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
    }
}

void HostGraph::load_graph_mtx(char *graph_path) {
    if (try_load_csr_cache(graph_path, &total_vertices_, &total_edge_,
                           &host_offset_, &host_edge_, &host_edge_weight_)) {
        std::cout << "[cache] CSR read from " << csr_cache_path(graph_path)
                  << " (skipped .mtx parse)" << std::endl;
        return;
    }
    if(loadMMSparseMatrix(graph_path, 'd', true, &total_vertices_, &total_vertices_, &total_edge_,
                                 &host_edge_weight_, &host_offset_, &host_edge_, true)){
        exit(EXIT_FAILURE);
    }
    write_csr_cache(graph_path, total_vertices_, total_edge_, host_offset_, host_edge_);
}

void HostGraph::randomly_generate_graph(int random_vertex_num, double sparsity) {
    int k = 0;
    int l = 0;
    weight_t *mat = new weight_t[random_vertex_num * random_vertex_num];
    memset(mat, 0, random_vertex_num * random_vertex_num * sizeof(weight_t));
    for(int i = 0; i < random_vertex_num; i++)
    {
        for(int j=0; j < random_vertex_num; j++)
        {
            size_t x = rand() % 1000000;
            if( x < 1000000.0 * sparsity )
            {
                mat[i * random_vertex_num + j] = x / 1000000.0 + 1.0;
                total_edge_++;
            }
        }
    }

    host_offset_ = new vertex_t[total_vertices_ + 1];
    host_edge_ = new edge_t[total_edge_];
    host_edge_weight_ = new weight_t[total_edge_];

    for(int i = 0; i < random_vertex_num; i++)
    {
        for(int j = 0; j < random_vertex_num; j++)
        {
            if(j == 0)
            {
                host_offset_[l++] = k;
            }
            if(mat[i * random_vertex_num + j] != 0)
            {
                host_edge_[k] = j;
                host_edge_weight_[k] = mat[i * random_vertex_num + j];
                k++;
            }
        }
    }
    host_offset_[l] = total_edge_;
}


vertex_t HostGraph::get_total_vertices_() {
    return total_vertices_;
}

edge_t HostGraph::get_total_edge_() {
    return total_edge_;
}

void HostGraph::set_total_vertices_(vertex_t total_vertices) {
    total_vertices_ = total_vertices;
}

void HostGraph::set_total_edge_(vertex_t total_edge) {
    total_edge_ = total_edge;
}

vertex_t *HostGraph::get_host_offset_() {
    return host_offset_;
}

edge_t *HostGraph::get_host_edge_() {
    return host_edge_;
}

weight_t *HostGraph::get_host_edge_weight_() {
    return host_edge_weight_;
}

double HostGraph::get_mass_() {
    return mass_;
}

void HostGraph::compute_total_edge_weight() {
    mass_ = thrust::reduce(host_edge_weight_, host_edge_weight_ + total_edge_);
//    printf("mass: %f\n", mass_);
}