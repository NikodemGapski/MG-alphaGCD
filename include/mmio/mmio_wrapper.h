#ifndef MMIO_WRAPPER_H
#define MMIO_WRAPPER_H
#include "./mmio.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstring>
#include "../common.h"

int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, vertex_t *m, vertex_t *n, edge_t *nnz,
                       weight_t **aVal, vertex_t **aRowInd, edge_t **aColInd, int extendSymMatrix);

// Directed loader: builds separate out-CSR and in-CSR (transpose) from a
// general (directed) MTX file.  On return, nnz_out == nnz_in == number of
// directed edges in the file.
int loadMMDirectedSparseMatrix(
    char *filename,
    vertex_t *m,
    // out-CSR
    vertex_t **out_offset, edge_t **out_edge, weight_t **out_weight, edge_t *nnz_out,
    // in-CSR (transpose)
    vertex_t **in_offset,  edge_t **in_edge,  weight_t **in_weight,  edge_t *nnz_in,
    // per-vertex out-strength s_out[v] = sum of outgoing edge weights
    weight_t **s_out
);

#endif //MMIO_WRAPPER_H
