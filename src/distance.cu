#include <iostream>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "distance.h"
#include "atom.cuh"

using namespace std;

inline cudaError_t checkCuda(cudaError_t result, int line )
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s at line %d\n", cudaGetErrorString(result),line);
  }
  return result;
}


__global__ void compute_total_distance_kernel_non_matrix(double* res, double* distances, int number_of_atoms, double* sdistance){

    
    
    extern __shared__ double tmp[];

    uint tid = threadIdx.x;
    //uint i = threadIdx.x  + blockIdx.x*blockDim.x;
    uint i = threadIdx.x + blockIdx.x*number_of_atoms;
    //if(i <  number_of_atoms*number_of_atoms*(blockIdx.x+1)){
    if(i < (blockIdx.x+1)*number_of_atoms){
        tmp[tid] = distances[i];
        //printf(" Blockid %d sdistance %lf of %d distance %lf\n",blockIdx.x, tmp[tid],tid,distances[i]);
    }
    else 
        tmp[tid] = 0;
        
    
    __syncthreads();

    

    for(unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
        
        if(tid < s){
            tmp[tid] += tmp[tid+s];
        }
        __syncthreads();
        //cg::sync(cta);
    }
    
    __syncthreads();
    //printf("%d %f\n", tid, tmp[tid]);

    if(tid == 0){
        res[blockIdx.x] = tmp[0];
        //printf("Sum: %lf  blockIdx: %d\n",res[blockIdx.x],blockIdx.x);
        //printf("blockDim size: %d\n", blockDim.x);
    }
    return;
}

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
        //if(blockIdx.x == 63 ) printf("%d %f atom id %d %lf x %lf y %lf z\n",
         //tid, res[tid],atoms[tid].id, atoms[tid].position.x,atoms[tid].position.y,atoms[tid].position.z);
    }
    //printf("point distance of %d : %lf\n", tid, res[tid]);
}


double* distance_v3(vector<atom_st> atoms, int number_of_atoms, int num_of_block){

    cudaError_t err;

    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    int deviceId;

    double* d_distance;
    
    
    checkCuda( cudaMalloc(&d_distance, 2*num_of_block* number_of_atoms * number_of_atoms*sizeof(double)), __LINE__);
    
    atom_st* atoms_tmp = (atom_st*)malloc(num_of_block*size_of_atoms);
    double* res;
    
    cudaGetDevice(&deviceId);
    checkCuda( cudaMallocManaged(&atoms_tmp, num_of_block * size_of_atoms), __LINE__);
    
    cudaMallocManaged(&res, num_of_block * sizeof(double));
    
    for(int i = 0; i < number_of_atoms * num_of_block; i++){
        atoms_tmp[i] = atoms[i];
    }

    atom_st * d_atoms;

    checkCuda( cudaMalloc(&d_atoms,size_of_atoms*num_of_block), __LINE__);

    checkCuda( cudaMemcpy(d_atoms, atoms_tmp, size_of_atoms*num_of_block, cudaMemcpyHostToDevice), __LINE__);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
   
    cudaMemPrefetchAsync(res, num_of_block*sizeof(double), deviceId);

    double* sdistance;
    checkCuda( cudaMalloc(&sdistance, 2* num_of_block * number_of_atoms*number_of_atoms*sizeof(double)) ,__LINE__);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    
    compute_point_distance_non_matrix<<<num_of_block, 512>>>(d_distance, d_atoms, number_of_atoms);
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s at %d\n", cudaGetErrorString(err),__LINE__);
    }

    cudaDeviceSynchronize();

    cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
    
    compute_total_distance_kernel_non_matrix<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);



    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
    }

    cudaDeviceSynchronize();
    
    
    
    checkCuda ( cudaFree(d_distance),__LINE__);
    checkCuda ( cudaFree(sdistance), __LINE__);
    
    checkCuda ( cudaFree(d_atoms), __LINE__);
    
    return res;
}

