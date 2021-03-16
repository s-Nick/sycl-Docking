#ifndef ATOM_H_
#define ATOM_H_


#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

/**
 * Struct used to define the structure of each atom keeping only the 
 * useful information: id and position.
 **/
struct atom_st
{
    unsigned int id;
    double3 position;
};

#endif