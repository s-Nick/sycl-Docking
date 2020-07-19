#include <iostream>
#include <vector>

#include "cooperative_groups.h"
#include "cooperative_groups_helpers.h"

#include "cuda_runtime.h"
#include "cuda.h"

#include "distance.h"
#include "atom.cuh"

using namespace std;

namespace cg = cooperative_groups;

__global__ void compute_total_distance_kernel(double* res, double* distances, int number_of_atoms){

    //cg::thread_block cta = cg::this_thread_block();
    __shared__ double sdistance[100];

    uint tid = threadIdx.x;
    uint i = threadIdx.x  + blockIdx.x*blockDim.x;
    sdistance[tid] = (i < number_of_atoms) ? distances[i] : 0;
    
    //cg::sync(cta);
    __syncthreads();

    //printf("%d %f\n", tid, sdistance[tid]);

    for(uint s=1; s < blockDim.x;  s*=2){
        if(tid % (2*s) == 0){
            sdistance[tid] += sdistance[tid+s];
        }
        __syncthreads();
        //cg::sync(cta);
    }
    
    if(tid == 0){
        res[blockIdx.x] = sdistance[0];
        //printf("Sum: %lf  blockIdx: %d\n",res[blockIdx.x],blockIdx.x);
    }
}

__global__ void compute_point_distance_kernel(double* res, atom_st* atoms, int num_of_atoms ){
    
    int tid = threadIdx.x + blockIdx.x*gridDim.x;
    res[tid] = 0;
    if(tid < num_of_atoms){
        double dx,dy,dz;
        for(int j = 0; j < num_of_atoms; j++){
            dx = atoms[tid].position.x - atoms[j].position.x;
            dy = atoms[tid].position.y - atoms[j].position.y;
            dz = atoms[tid].position.z - atoms[j].position.z;
            double distSqr = dx*dx + dy*dy + dz*dz;
            res[tid] += sqrt(distSqr);
        }
    }
}

double distance(vector<atom_st> atoms, int number_of_atoms){

    cudaError_t err;

    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    int deviceId;

    double* distances;
    /*
    float* x_dist;
    float* y_dist;
    float* z_dist;
*/
    atom_st* atoms_tmp;
    double*res;
    
    cudaGetDevice(&deviceId);
    cudaMallocManaged(&atoms_tmp,size_of_atoms);
    cudaMallocManaged(&distances,number_of_atoms*sizeof(double));
    cudaMallocManaged(&res, 1 * sizeof(double));
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    //cudaMallocManaged(&x_dist,number_of_atoms*number_of_atoms*sizeof(float));
    //cudaMallocManaged(&y_dist,number_of_atoms*number_of_atoms*sizeof(float));
    //cudaMallocManaged(&z_dist,number_of_atoms*number_of_atoms*sizeof(float));

    cudaMemPrefetchAsync(distances, number_of_atoms*sizeof(double), deviceId);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    //cudaMemPrefetchAsync(x_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);
    //cudaMemPrefetchAsync(y_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);
    //cudaMemPrefetchAsync(z_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);

    int number_of_blocks = 32;
    int nThreads = 256;
    
    for(int i = 0; i < number_of_atoms; i++){
        atoms_tmp[i] = atoms[i];
        //printf("atom %d %lf x %lf y %lf z\n",i, atoms_tmp[i].position.x,atoms_tmp[i].position.y,atoms_tmp[i].position.z);
    }
    res[0] = 0;

    compute_point_distance_kernel<<<1, number_of_atoms>>>(distances, atoms_tmp, number_of_atoms);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    /*
    float tmp=0;
    
    for(int i = 0; i < number_of_atoms; i++){
        cout << i << " " << distances[i] << endl;
        tmp += distances[i];
    }
    cout << endl;// , number_of_atoms*sizeof(float)
    */

    compute_total_distance_kernel<<<1, number_of_atoms>>>(res, distances, number_of_atoms);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
    }
    cudaDeviceSynchronize();
    
    //cout << "Total: " << res[0] << " " << __LINE__ << endl;

    cudaFree(atoms_tmp);
    cudaFree(distances);

    double ret  = *res;
    cudaFree(res);
    
    return ret;
}

