#ifndef QUATERNION_H_
#define QUATERNION_H_

#include<vector>
#include "atom.cuh"

class Rotation{
    
    public:
        
        std::vector<std::vector<atom_st>> rotate_v5(int angle, std::vector<atom_st>& atoms,double3& passingPoint,double4* unit_quaternion);


        
    };

#endif