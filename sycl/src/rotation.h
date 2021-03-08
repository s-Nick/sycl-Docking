#ifndef ROTATION_H_
#define ROTATION_H_

#include <vector>
#include "atom.h"

class Rotation
{

public:
    std::vector<std::vector<atom_st>> rotate(int angle, std::vector<atom_st> &atoms, cl::sycl::double3 &passingPoint, cl::sycl::double4 *unit_quaternion);
};

#endif