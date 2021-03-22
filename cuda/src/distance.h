#ifndef DISTANCE_H_
#define DISTANCE_H_

#include <vector>
#include "atom.cuh"

using namespace std;

double* distance_v3(vector<atom_st> atoms, int num, int num_of_block);


#endif