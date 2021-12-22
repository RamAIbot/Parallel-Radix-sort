#ifndef SORT_H
#define SORT_H

#include "edgelist.h"
#include "graph.h"

#include <stdbool.h>

// Order edges by id of a source vertex,
// using the Counting Sort
// Complexity: O(E + V)

#ifdef OPENMP_HARNESS
struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph);
#endif

#ifdef MPI_HARNESS
struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph,int argc, char **argv);
#endif

#ifdef HYBRID_HARNESS
struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph);
#endif

struct Graph *countSortEdgesBySource (struct Graph *graph);

extern int numThreads;

#endif