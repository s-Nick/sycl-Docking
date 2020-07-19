#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "atom.cuh"
#include <vector>


using namespace std;

//double distance(atom_st* atoms, int num);
double distance(vector<atom_st> atoms, int num);


#undef DISTANCE_H_
#endif