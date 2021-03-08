#ifndef ATOM_H_
#define ATOM_H_

#include "helper.h"

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