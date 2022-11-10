#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <vector>
#include "atom.cuh"

double* distance_v3(std::vector<atom_st> atoms, int num, int num_of_block);


#endif