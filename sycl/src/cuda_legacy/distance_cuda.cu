#include <iostream>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "distance.h"
#include "atom_cuda.cuh"

using namespace std;

inline cudaError_t checkCuda(cudaError_t result, int line )
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s at line %d\n", cudaGetErrorString(result),line);
  }
  return result;
}

/**
 * Compute the total internal distance of each rotation. Each block take care of one angle of the rotation and 
 * store the result in an array. The position in the array corresponds to the angle of the rotation in degrees.
 * 
 * @param res Array that store the final results.
 * @param distances Array with the point distances of each atom of the molecule.
 * @param number_of_atoms number of the atoms in the molecule. 
 **/
__global__ void compute_total_distance_kernel_non_matrix(double* res, double* distances, int number_of_atoms){
    
    extern __shared__ double tmp[];

    uint tid = threadIdx.x;

    uint i = threadIdx.x + blockIdx.x*number_of_atoms;
    
    if(i < (blockIdx.x+1)*number_of_atoms){
        tmp[tid] = distances[i];
    }
    else 
        tmp[tid] = 0;
        
    __syncthreads();
 
    for(unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
        
        if(tid < s){
            tmp[tid] += tmp[tid+s];
        }
        __syncthreads();
    }
    
    __syncthreads();
    
    if(tid == 0){
        res[blockIdx.x] = tmp[0];
    }
    return;
}


/**
 * Compute the  Eucledian distance of each atom with all the others. Each block compute the result of a Rotation.
 * 
 * @param res store the result.
 * @param atoms All the atoms of all the rotations to take into account.
 * @param num_of_atoms number of atoms of the molecule.
 **/
__global__ void compute_point_distance_non_matrix(double* res, atom_st* atoms, int num_of_atoms ){
    
    int tid = threadIdx.x + blockIdx.x*num_of_atoms;
    res[tid] = 0;
    if(tid < num_of_atoms*(blockIdx.x+1)){
        double dx,dy,dz,distSqr;
        for(int j = num_of_atoms*blockIdx.x; j < num_of_atoms*(blockIdx.x+1); j++){
            dx = atoms[tid].position.x - atoms[j].position.x;
            dy = atoms[tid].position.y - atoms[j].position.y;
            dz = atoms[tid].position.z - atoms[j].position.z;
            distSqr = dx*dx + dy*dy + dz*dz;
            res[tid] += sqrt(distSqr);
        }
    }
}

/**
 * Compute the internal distance of the molecule. It calls two kernels, one for the point distance of each atom 
 * with the others, and the second to sum all the distance of each atom. It is possible to compute all the distance 
 * of all the rotation all together calling the kernels with 360 blocks. Each result will be stored in the angle corresponding position.
 * 
 * @param atoms All the atoms of all the rotation computed.
 * @param number_of_atoms number of atoms in the molecule
 * @param num_of_block numbre of block used in the rotation.
 **/

double* distance_v3(vector<atom_st> atoms, int number_of_atoms, int num_of_block){

    cudaError_t err;

    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    int deviceId;
    double* d_distance;
    atom_st* atoms_tmp = (atom_st*)malloc(num_of_block*size_of_atoms);
    double* res;
    atom_st * d_atoms;

    checkCuda( cudaMalloc(&d_distance, 2*num_of_block* number_of_atoms * number_of_atoms*sizeof(double)), __LINE__);
    
    checkCuda( cudaGetDevice(&deviceId), __LINE__);
    checkCuda( cudaMallocManaged(&atoms_tmp, num_of_block * size_of_atoms), __LINE__);
    
    checkCuda( cudaMallocManaged(&res, num_of_block * sizeof(double)), __LINE__);
    
    for(int i = 0; i < number_of_atoms * num_of_block; i++){
        atoms_tmp[i] = atoms[i];
    }

    checkCuda( cudaMalloc(&d_atoms,size_of_atoms*num_of_block), __LINE__);

    checkCuda( cudaMemcpy(d_atoms, atoms_tmp, size_of_atoms*num_of_block, cudaMemcpyHostToDevice), __LINE__);
   
    checkCuda( cudaMemPrefetchAsync(res, num_of_block*sizeof(double), deviceId), __LINE__);
    
    compute_point_distance_non_matrix<<<num_of_block, 512>>>(d_distance, d_atoms, number_of_atoms);
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s at %d\n", cudaGetErrorString(err),__LINE__);
    }

    cudaDeviceSynchronize();

    checkCuda( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ), __LINE__);
    
    compute_total_distance_kernel_non_matrix<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
    }

    cudaDeviceSynchronize();
    
    checkCuda ( cudaFree(d_distance),__LINE__);
    
    checkCuda ( cudaFree(d_atoms), __LINE__);
    
    return res;
}