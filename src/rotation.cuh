#ifndef QUATERNION_H_
#define QUATERNION_H_

#include<vector>
#include "cublas_v2.h"
#include "atom.cuh"

class Rotation{
    
    public:
        
        Rotation(double3 vector);
        __host__ void getQuaternion(double4* h_quat,double4* unit_quaternion);
        //__host__ std::vector<std::vector<double>> getRotationMatrix();
        //~Quaternion();
        atom_st* rotate(double angle, std::vector<atom_st>& atoms,double3& passingPoint);

        atom_st* rotate_v2(int angle, std::vector<atom_st>& atoms,double3& passingPoint,double4 unit_quaternion);

        std::vector<atom_st*> rotate_v3(int angle, std::vector<atom_st>& atoms,double3& passingPoint,double4* unit_quaternion);

        std::vector<std::vector<atom_st>> rotate_v4(int angle, std::vector<atom_st>& atoms,double3& passingPoint,double4* unit_quaternion);


    private:
        double4 quaternion;
        double4 unit_quaternion;
        
        std::vector<std::vector<double>> rotation_matrix;
        
        void normalize(double* quat,double* norm);//float4& vec, float *norm

    };

#endif