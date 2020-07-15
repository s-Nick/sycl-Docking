#ifndef QUATERNION_H_
#define QUATERNION_H_

#include<vector>
#include "cublas_v2.h"
#include "atom.cuh"

class Rotation{
    
    public:
        
        Rotation(float3 vector);
        __host__ void getQuaternion(float4* h_quat,float4* unit_quaternion);
        //__host__ std::vector<std::vector<double>> getRotationMatrix();
        //~Quaternion();
        atom_st* rotate(float angle, std::vector<atom_st>& atoms,float3& passingPoint);
        

    private:
        float4 quaternion;
        float4 unit_quaternion;
        
        std::vector<std::vector<double>> rotation_matrix;
        
        void normalize(float* quat,float* norm);//float4& vec, float *norm

    };

#endif