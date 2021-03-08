#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "atom_cuda.cuh"
#include <vector>


using namespace std;

double* distance_v3(vector<atom_st> atoms, int num, int num_of_block);

#undef DISTANCE_H_
#endif