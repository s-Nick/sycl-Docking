#include "rotation.cuh"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include <stdio.h>
#include "atom.cuh"
#include <iostream>
#include "math_constants.h"

#define NUM_OF_STREAMS 360

using namespace std;

inline cudaError_t checkCuda(cudaError_t result, int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s at line %d\n", cudaGetErrorString(result),line);
  }
  return result;
}


__device__ void initFirstRow_v2(double rotation_matrix[3][3], double4 & unit_quat){
    
    
    
    rotation_matrix[0][0] =  1-2*(pow(unit_quat.y,2) + pow(unit_quat.z,2));
    
    rotation_matrix[0][1] = 2*(unit_quat.x*unit_quat.y - unit_quat.z*unit_quat.w);

    rotation_matrix[0][2] = 2*(unit_quat.x*unit_quat.z + unit_quat.y*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(rotation_matrix[0][i] == -0) rotation_matrix[0][i] = 0;
    }
    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);


}

__device__ void initSecondRow_v2(double rotation_matrix[3][3], double4 & unit_quat){
    
    
    rotation_matrix[1][0] =  2*(unit_quat.x*unit_quat.y + unit_quat.z*unit_quat.w);

    rotation_matrix[1][1] = 1-2*(unit_quat.x*unit_quat.x + pow(unit_quat.z,2));//unit_quat.z*unit_quat.z);

    rotation_matrix[1][2] = 2*(unit_quat.y*unit_quat.z - unit_quat.x*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(rotation_matrix[1][i] == -0) rotation_matrix[1][i] = 0;
    }

    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);

}


__device__ void initThirdRow_v2(double rotation_matrix[3][3], double4 & unit_quat){
    
    
    rotation_matrix[2][0] =  2*(unit_quat.x*unit_quat.z - unit_quat.y*unit_quat.w);

    rotation_matrix[2][1] = 2*(unit_quat.y*unit_quat.z + unit_quat.x*unit_quat.w);

    rotation_matrix[2][2] = 1-2*(pow(unit_quat.x,2) + pow(unit_quat.y,2));
    for(int i= 0;i < 3;i++){
        if(rotation_matrix[2][i] == -0) rotation_matrix[2][i] = 0;
    }
    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);

}



__global__ void rotation_kernel_v5(atom_st* res, atom_st* atoms,
                                int number_of_atoms, double3 pp, double4* unit_quaternion, int s){
    

    int tid = threadIdx.x; //+ blockIdx.x*blockDim.x;
    if(s+blockIdx.x < 360){
    //printf("unit_q %lf x %lf y %lf z %lf w\n", unit_quaternion[s].x,unit_quaternion[s].y,unit_quaternion[s].z,unit_quaternion[s].w);
        __shared__ double rot_matrix[3][3];
    
        if(tid == 0) initFirstRow_v2(rot_matrix,unit_quaternion[s+blockIdx.x]);
        else if(tid == 1) initSecondRow_v2(rot_matrix,unit_quaternion[s+blockIdx.x]);
        else if(tid == 2) initThirdRow_v2(rot_matrix,unit_quaternion[s+blockIdx.x]);

    
        __syncthreads();

        int index = threadIdx.x + blockIdx.x*number_of_atoms;
        if(index < number_of_atoms*(blockIdx.x+1) && number_of_atoms*blockIdx.x <= index){
            //printf("kernel_v5 %d block %d\n", index,blockIdx.x);
            res[index].id = atoms[tid].id;

            res[index].position.x = atoms[tid].position.x * rot_matrix[0][0] + \
                                atoms[tid].position.y * rot_matrix[0][1] + \
                                atoms[tid].position.z * rot_matrix[0][2] + pp.x;
        
            res[index].position.y = atoms[tid].position.x * rot_matrix[1][0] + \
                                atoms[tid].position.y * rot_matrix[1][1] + \
                                atoms[tid].position.z * rot_matrix[1][2] + pp.y;
            
            res[index].position.z = atoms[tid].position.x * rot_matrix[2][0] + \
                                atoms[tid].position.y * rot_matrix[2][1] + \
                                atoms[tid].position.z * rot_matrix[2][2] + pp.z;
        }
        
    }

}


__global__ void first_translation(atom_st* atoms,double3 pp,int number_of_atoms){

    int tidx = threadIdx.x;
     
    if(tidx < number_of_atoms){
        atoms[tidx].position.x -= pp.x;
        atoms[tidx].position.y -= pp.y;
        atoms[tidx].position.z -= pp.z;
    }

}

__global__ void back_translation(atom_st* atoms,double3 pp,int number_of_atoms){

    int tidx = threadIdx.x;
     
    if(tidx < number_of_atoms){
        atoms[tidx].position.x += pp.x;
        atoms[tidx].position.y += pp.y;
        atoms[tidx].position.z += pp.z;
    }

}



vector<vector<atom_st>> Rotation::rotate_v5(int angle, std::vector<atom_st>& atoms_st, double3& pp, double4* unit_quaternion){

    int deviceId;
    int number_of_atoms = atoms_st.size();
    int size_of_atoms = number_of_atoms*sizeof(atom_st);

    atom_st *atoms;    
    cudaError_t err;

    atom_st * h_res;
    atom_st * d_res;    


    checkCuda( cudaGetDevice(&deviceId), __LINE__ );
    cudaMallocManaged(&atoms, size_of_atoms);
    
    checkCuda( cudaMallocHost(&h_res, size_of_atoms*NUM_OF_STREAMS),__LINE__);
    
    checkCuda( cudaMalloc(&d_res,size_of_atoms*NUM_OF_STREAMS), __LINE__);

    
    //initialize vector of atoms
    int i = 0;
    for(auto at : atoms_st){
        atoms[i] = at;
        i++;
    }
    
    cudaMemPrefetchAsync(atoms,size_of_atoms, deviceId);
    
    double3 passingPoint = pp;
    
    first_translation<<<1,number_of_atoms>>>(atoms,passingPoint, number_of_atoms);

    cudaDeviceSynchronize();
    checkCuda( cudaMemPrefetchAsync(unit_quaternion,2*360*sizeof(double4), deviceId) ,__LINE__);
    checkCuda( cudaMemPrefetchAsync(atoms,size_of_atoms,deviceId), __LINE__);
    
    rotation_kernel_v5<<<NUM_OF_STREAMS,64,0>>>(d_res,atoms,number_of_atoms,passingPoint,unit_quaternion,angle);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        cout << __LINE__ << endl;
        printf("Error %s \n", cudaGetErrorString(err));
    }    

    checkCuda( cudaDeviceSynchronize(),__LINE__);
    
    checkCuda( cudaMemcpy(h_res, d_res, size_of_atoms * NUM_OF_STREAMS, cudaMemcpyDeviceToHost), __LINE__ );
    
    checkCuda( cudaFree(atoms), __LINE__ );
    
    vector<vector<atom_st>> result_to_return;

    vector<atom_st> tmp;
    for(int i = 0; i < NUM_OF_STREAMS; i++ ){
        for(int c = atoms_st.size()*i; c < atoms_st.size()*(i+1); c++){
            tmp.push_back(h_res[c]);
        }
        result_to_return.push_back(tmp);
        tmp.clear();
    }

    checkCuda( cudaFreeHost(h_res), __LINE__);
    checkCuda( cudaFree(d_res),__LINE__ );


    return result_to_return;
}
