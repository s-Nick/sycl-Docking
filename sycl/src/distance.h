#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "atom.h"
#include <vector>

using namespace std;

double *distance(vector<atom_st> atoms, int num, int num_of_block, cl::sycl::queue& queue_gpu);

#undef DISTANCE_H_
#endif