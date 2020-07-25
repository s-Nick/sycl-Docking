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

inline cudaError_t checkCuda(cudaError_t result, int line )
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s at line %d\n", cudaGetErrorString(result),line);
  }
  return result;
}

__global__ void compute_total_distance_kernel(double* res, double* distances, int number_of_atoms){

    //cg::thread_block cta = cg::this_thread_block();
    
    extern __shared__ double sdistance[];

    uint tid = threadIdx.x;
    uint i = threadIdx.x  + blockIdx.x*blockDim.x;
    sdistance[tid] = (i < number_of_atoms*number_of_atoms) ? distances[i] : 0;
    
    //printf("sdistance %lf of %d\n", sdistance[i],tid);

    //cg::sync(cta);
    __syncthreads();

    

    for(uint s=1; s < blockDim.x;  s*=2){
        if(tid % (2*s) == 0){
            sdistance[tid] += sdistance[tid+s];
        }
        __syncthreads();
        //cg::sync(cta);
    }
    __syncthreads();
    
    if(tid == 0){
        printf("%d %f\n", tid, sdistance[tid]);
        res[blockIdx.x] = sdistance[0];
        printf("Sum: %lf  blockIdx: %d\n",res[0],blockIdx.x);
    }
}

__global__ void compute_total_distance_kernel_v2(double* res, double* distances, int number_of_atoms, double* sdistance){

    //cg::thread_block cta = cg::this_thread_block();
    
    extern __shared__ double tmp[];

    uint tid = threadIdx.x;
    uint i = threadIdx.x  + blockIdx.x*blockDim.x;
    //if(i <  number_of_atoms*number_of_atoms*(blockIdx.x+1)){
    tmp[tid] = distances[i];
    
    
    
    printf("sdistance %lf of %d distance %lf\n", tmp[i],tid,distances[i]);

    //cg::sync(cta);
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
        printf("blockDim size: %d\n", blockDim.x);
    }
}

__global__ void compute_total_distance_kernel_v3(double* res, double* distances, int number_of_atoms, double* sdistance){

    //cg::thread_block cta = cg::this_thread_block();
    
    extern __shared__ double tmp[];

    uint tid = threadIdx.x;
    //uint i = threadIdx.x  + blockIdx.x*blockDim.x;
    uint i = threadIdx.x + blockIdx.x*number_of_atoms*number_of_atoms;
    //if(i <  number_of_atoms*number_of_atoms*(blockIdx.x+1)){
    if(i < (blockIdx.x+1)*number_of_atoms*number_of_atoms){
        tmp[tid] = distances[i];
        //printf(" Blockid %d sdistance %lf of %d distance %lf\n",blockIdx.x, tmp[tid],tid,distances[i]);
    }
    else 
        tmp[tid] = 0;
        
    //cg::sync(cta);
    __syncthreads();

    //if(blockIdx.x == 7) printf("sdistance %lf of %d distance %lf\n", tmp[tid],tid,distances[i]);

    for(unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
        
        if(tid < s){
            tmp[tid] += tmp[tid+s];
            //if(blockIdx.x == 3) printf("%d %f\n", tid, tmp[tid]);
        }
        __syncthreads();
        //cg::sync(cta);
    }
    
    __syncthreads();
    

    if(tid == 0){
        res[blockIdx.x] = tmp[0];
        //printf("Sum: %lf  blockIdx: %d\n",res[blockIdx.x],blockIdx.x);
        //printf("blockDim size: %d\n", blockDim.x);
    }
    return;
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


__global__ void compute_point_distance_kernel(double* res, atom_st* atoms, int num_of_atoms ){
    
    int tid = threadIdx.x + blockIdx.x*gridDim.x;
    //res[tid] = 0;
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
    //printf("point distance of %d : %lf\n", tid, res[tid]);
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

__global__ void compute_point_distance_kernel_v2(double* res, atom_st* atoms, int num_of_atoms ){
    
    int tidx = threadIdx.x + blockIdx.x*gridDim.x;
    int tidy = threadIdx.y + blockIdx.y*gridDim.y;
    int index = tidx + tidy*num_of_atoms;
    res[tidx] = 0;
    if(tidx < num_of_atoms && tidy < num_of_atoms){
        double dx,dy,dz;
        for(int j = 0; j < num_of_atoms; j++){
            dx = atoms[tidx].position.x - atoms[tidy].position.x;
            dy = atoms[tidx].position.y - atoms[tidy].position.y;
            dz = atoms[tidx].position.z - atoms[tidy].position.z;
            double distSqr = dx*dx + dy*dy + dz*dz;
            res[index] = sqrt(distSqr);
        }
    }
    //printf("point distance of %d  %d: %lf\n", tidx, tidy, res[index]);
}

__global__ void compute_point_distance_kernel_v3(double* res, atom_st* atoms, int num_of_atoms ){
    
    int tidx = threadIdx.x + blockIdx.x*num_of_atoms;
    int tidy = threadIdx.y + blockIdx.y*num_of_atoms;
    int index = tidx + tidy*num_of_atoms - blockIdx.x*num_of_atoms - blockIdx.y*num_of_atoms;//+ blockIdx.x*num_of_atoms*num_of_atoms - blockIdx.x*num_of_atoms;
    //res[index] = 0;
    //if( num_of_atoms*(blockIdx.x) <= tidx < num_of_atoms*(blockIdx.x+1) && num_of_atoms*blockIdx.y <= tidy < num_of_atoms*(blockIdx.y+1)){
    if( tidx < num_of_atoms*(blockIdx.x+1) &&  tidy < num_of_atoms*(blockIdx.y+1)){
        double dx,dy,dz;
        dx = atoms[tidx].position.x - atoms[tidy].position.x;
        dy = atoms[tidx].position.y - atoms[tidy].position.y;
        dz = atoms[tidx].position.z - atoms[tidy].position.z;
        //double distSqr = pow(dx,2) + pow(dy,2) + pow(dz,2);
        double distSqr = dx*dx + dy*dy + dz*dz;
        res[index] = sqrt(distSqr);
    }
    //printf("index %d\n",index);
    //if(tidx == 0) printf( "pos 13 %lf x %lf y %lf z pos 33 %lf x %lf y %lf z\n", atoms[1].position.x,atoms[1].position.y,atoms[1].position.z,\
    //atoms[21].position.x,atoms[21].position.y,atoms[21].position.z);
    //printf(" BlockId %d point distance of %d  %d: %lf\n",blockIdx.x ,tidx, tidy, res[index]);
}

__global__ void test_total(double* res, atom_st* atoms, int num_of_atoms){
    
    extern __shared__ double tmp[];

    int tidx = threadIdx.x + blockIdx.x*num_of_atoms;
    int tidy = threadIdx.y + blockIdx.y*num_of_atoms;
    int index = threadIdx.x + threadIdx.y*num_of_atoms;
    if( tidx < num_of_atoms*(blockIdx.x+1) && tidy < num_of_atoms*(blockIdx.y+1) ){
        double dx,dy,dz;
        dx = atoms[tidx].position.x - atoms[tidy].position.x;
        dy = atoms[tidx].position.y - atoms[tidy].position.y;
        dz = atoms[tidx].position.z - atoms[tidy].position.z;
        double distSqr = pow(dx,2) + pow(dy,2) + pow(dz,2);
        tmp[index] = sqrt(distSqr);
    }
    else{
        tmp[index] = 0;
    }
    
    
    __syncthreads();

    uint c = threadIdx.x;
    printf("tidx %d distance %lf\n", index,tmp[index]);
    //printf("index %d\n",index);
    for(unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
        
        if(c < s){
            tmp[c] += tmp[c+s];
        }
        __syncthreads();
        //cg::sync(cta);
    }
    
    __syncthreads();
    //printf("%d %f\n", tid, tmp[tid]);

    if(c == 0){
        res[blockIdx.x] = tmp[0];
        //printf("Sum: %lf  blockIdx: %d\n",res[blockIdx.x],blockIdx.x);
        printf("blockDim size: %d %d\n", blockDim.x,blockDim.y);
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
    
    //cudaMallocManaged(&distances,number_of_atoms*sizeof(double));
    checkCuda( cudaMalloc(&distances,number_of_atoms*sizeof(double)), __LINE__ );
    
    cudaMallocManaged(&res, 1 * sizeof(double));
    
    for(int i = 0; i < number_of_atoms; i++){
        atoms_tmp[i] = atoms[i];
        //printf("atom %d %lf x %lf y %lf z\n",i, atoms_tmp[i].position.x,atoms_tmp[i].position.y,atoms_tmp[i].position.z);
    }


    res[0] = 0;

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    //cudaMallocManaged(&x_dist,number_of_atoms*number_of_atoms*sizeof(float));
    //cudaMallocManaged(&y_dist,number_of_atoms*number_of_atoms*sizeof(float));
    //cudaMallocManaged(&z_dist,number_of_atoms*number_of_atoms*sizeof(float));

    //cudaMemPrefetchAsync(distances, number_of_atoms*sizeof(double), deviceId);
    cudaMemPrefetchAsync(atoms_tmp, size_of_atoms, deviceId);
    cudaMemPrefetchAsync(res, 1*sizeof(double), deviceId);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    //cudaMemPrefetchAsync(x_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);
    //cudaMemPrefetchAsync(y_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);
    //cudaMemPrefetchAsync(z_dist, number_of_atoms*number_of_atoms*sizeof(float), deviceId);

    int number_of_blocks = 32;
    int nThreads = 256;
    
    
    

    compute_point_distance_kernel<<<1, 256>>>(distances, atoms_tmp, number_of_atoms);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    

    compute_total_distance_kernel<<<1, number_of_atoms+10,number_of_atoms*2*sizeof(double)>>>(res, distances, number_of_atoms);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
    }
    cudaDeviceSynchronize();
    
    //cout << "Total: " << res[0] << " " << __LINE__ << endl;
    //cudaMemPrefetchAsync(res, 1*sizeof(double), cudaCpuDeviceId);
    cudaFree(atoms_tmp);
    cudaFree(distances);

    double ret  = *res;
    cudaFree(res);
    
    return ret;
}

double distance_v2(vector<atom_st> atoms, int number_of_atoms){

    cudaError_t err;

    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    int deviceId;

    double* d_distance;
    //double* h_distance[number_of_atoms];
    /*
    for(int n = 0; n < number_of_atoms; n++){
        checkCuda( cudaMallocHost(&h_distance[n],number_of_atoms*sizeof(double)),__LINE__ );
    }
    */
    checkCuda( cudaMalloc(&d_distance, number_of_atoms * number_of_atoms*sizeof(double)), __LINE__);
    
    

    atom_st* atoms_tmp;
    double*res;
    
    cudaGetDevice(&deviceId);
    cudaMallocManaged(&atoms_tmp,size_of_atoms);
    
    //cudaMallocManaged(&distances,number_of_atoms*sizeof(double));
    //checkCuda( cudaMalloc(&distances,number_of_atoms*sizeof(double)) );
    
    cudaMallocManaged(&res, 1 * sizeof(double));
    
    for(int i = 0; i < number_of_atoms; i++){
        atoms_tmp[i] = atoms[i];
        //printf("atom %d %lf x %lf y %lf z\n",i, atoms_tmp[i].position.x,atoms_tmp[i].position.y,atoms_tmp[i].position.z);
    }


    res[0] = 0;

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
   
    cudaMemPrefetchAsync(atoms_tmp, size_of_atoms, deviceId);
    cudaMemPrefetchAsync(res, 1*sizeof(double), deviceId);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    
    int number_of_blocks = 32;
    int nThreads = 256;
    
    
    dim3 dimBlock(number_of_atoms,number_of_atoms);

    
    compute_point_distance_kernel_v2<<<1, dimBlock>>>(d_distance, atoms_tmp, number_of_atoms);
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    
    double* sdistance;
    checkCuda( cudaMalloc(&sdistance,number_of_atoms*number_of_atoms*sizeof(double)) ,__LINE__);
    
    //checkCuda( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), __LINE__);

//    compute_total_distance_kernel<<<1, number_of_atoms*number_of_atoms,3*number_of_atoms*number_of_atoms*sizeof(double)>>>(res, d_distance, number_of_atoms);

    compute_total_distance_kernel_v2<<<1, number_of_atoms*number_of_atoms,3*number_of_atoms*number_of_atoms*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
    }
    cudaDeviceSynchronize();

    //cout << "Total: " << res[0] << " " << __LINE__ << endl;
    //cudaMemPrefetchAsync(res, 1*sizeof(double), cudaCpuDeviceId);
    cudaFree(atoms_tmp);
    //cudaFree(distances);
    //for(int n =0 ; n < number_of_atoms; n++) checkCuda ( cudaFree(h_distance[n]),__LINE__);
    
    //for(int n =0 ; n < number_of_atoms; n++) checkCuda ( cudaFree(d_distance[n]),__LINE__);
    
    checkCuda ( cudaFree(d_distance),__LINE__);
    checkCuda (cudaFree(sdistance), __LINE__);
    //double ret  = *res;
    //cudaFree(res);
    
    return *res;
}

double* distance_v3(vector<atom_st> atoms, int number_of_atoms, int num_of_block){

    cudaError_t err;

    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    int deviceId;

    double* d_distance;
    //double* h_distance[number_of_atoms];
    /*
    for(int n = 0; n < number_of_atoms; n++){
        checkCuda( cudaMallocHost(&h_distance[n],number_of_atoms*sizeof(double)),__LINE__ );
    }
    */

    
    
    checkCuda( cudaMalloc(&d_distance, 2*num_of_block* number_of_atoms * number_of_atoms*sizeof(double)), __LINE__);
    
    atom_st* atoms_tmp = (atom_st*)malloc(num_of_block*size_of_atoms);
    double* res;
    
    cudaGetDevice(&deviceId);
    checkCuda( cudaMallocManaged(&atoms_tmp, num_of_block * size_of_atoms), __LINE__);
    
    //cudaMallocManaged(&distances,number_of_atoms*sizeof(double));
    //checkCuda( cudaMalloc(&distances,number_of_atoms*sizeof(double)) );
    
    cudaMallocManaged(&res, num_of_block * sizeof(double));
    
    for(int i = 0; i < number_of_atoms * num_of_block; i++){
        atoms_tmp[i] = atoms[i];
        //checkCuda( cudaMemcpy(&atoms_tmp[i], &atoms[i], size_of_atoms, cudaMemcpyHostToDevice) , __LINE__);
    }

    atom_st * d_atoms;

    checkCuda( cudaMalloc(&d_atoms,size_of_atoms*num_of_block), __LINE__);

    checkCuda( cudaMemcpy(d_atoms, atoms_tmp, size_of_atoms*num_of_block, cudaMemcpyHostToDevice), __LINE__);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
   
    //cudaMemPrefetchAsync(atoms_tmp, size_of_atoms * num_of_block, deviceId);
    cudaMemPrefetchAsync(res, num_of_block*sizeof(double), deviceId);

    double* sdistance;
    checkCuda( cudaMalloc(&sdistance, 2* num_of_block * number_of_atoms*number_of_atoms*sizeof(double)) ,__LINE__);

    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    /*
    if(number_of_atoms <= 32){
        dim3 dimBlock(number_of_atoms,number_of_atoms);

        //cout << "distance line " << __LINE__ << endl;
        //compute_point_distance_kernel_v3<<<num_of_block, dimBlock>>>(d_distance, d_atoms, number_of_atoms);
        //compute_point_distance_non_matrix<<<num_of_block, number_of_atoms>>>(d_distance, d_atoms, number_of_atoms);
        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s at %d\n", cudaGetErrorString(err),__LINE__);
        }

        cudaDeviceSynchronize();

        test_total<<<num_of_block,512, 2*number_of_atoms*number_of_atoms*sizeof(double)>>>(res,d_atoms,number_of_atoms);
        cudaDeviceSynchronize();
        //checkCuda( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), __LINE__);number_of_atoms * number_of_atoms

    //    compute_total_distance_kernel<<<1, number_of_atoms*number_of_atoms,3*number_of_atoms*number_of_atoms*sizeof(double)>>>(res, d_distance, number_of_atoms);
        //cout << "tot dist " << __LINE__ << endl;
        cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
        
        //compute_total_distance_kernel_v3<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);
        //compute_total_distance_kernel_non_matrix<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);
    

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
        }

    }
    else{
         
        compute_point_distance_non_matrix<<<num_of_block, number_of_atoms>>>(d_distance, d_atoms, number_of_atoms);
        
        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s at %d\n", cudaGetErrorString(err),__LINE__);
        }

        cudaDeviceSynchronize();

        //    double* sdistance;
        //checkCuda( cudaMalloc(&sdistance, 2* num_of_block * number_of_atoms*number_of_atoms*sizeof(double)) ,__LINE__);
        
        //checkCuda( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), __LINE__);number_of_atoms * number_of_atoms

    //    compute_total_distance_kernel<<<1, number_of_atoms*number_of_atoms,3*number_of_atoms*number_of_atoms*sizeof(double)>>>(res, d_distance, number_of_atoms);
        //cout << "tot dist " << __LINE__ << endl;
        cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
        
        compute_total_distance_kernel_non_matrix<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);

    

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
        }
    }*/

     
        compute_point_distance_non_matrix<<<num_of_block, 512>>>(d_distance, d_atoms, number_of_atoms);
        
        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s at %d\n", cudaGetErrorString(err),__LINE__);
        }

        cudaDeviceSynchronize();

        //    double* sdistance;
        //checkCuda( cudaMalloc(&sdistance, 2* num_of_block * number_of_atoms*number_of_atoms*sizeof(double)) ,__LINE__);
        
        //checkCuda( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), __LINE__);number_of_atoms * number_of_atoms

    //    compute_total_distance_kernel<<<1, number_of_atoms*number_of_atoms,3*number_of_atoms*number_of_atoms*sizeof(double)>>>(res, d_distance, number_of_atoms);
        //cout << "tot dist " << __LINE__ << endl;
        cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
        
        compute_total_distance_kernel_non_matrix<<<num_of_block, 512, 2*512*sizeof(double)>>>(res, d_distance, number_of_atoms,sdistance);

    

        err = cudaGetLastError();
        if(err != cudaSuccess){
            printf("Error %s %d\n", cudaGetErrorString(err), __LINE__);
        }

    cudaDeviceSynchronize();
    
    //for(int i = 0; i < num_of_block; i++) printf("Sum: %lf  blockIdx: %d\n",res[i],i);
    //cout << "Total: " << res[0] << " " << __LINE__ << endl;
    //cudaMemPrefetchAsync(res, 1*sizeof(double), cudaCpuDeviceId);
    //checkCuda( cudaFree(atoms_tmp), __LINE__);
    //cudaFree(distances);
    //for(int n =0 ; n < number_of_atoms; n++) checkCuda ( cudaFree(h_distance[n]),__LINE__);
    //free(atoms_tmp);
    //for(int n =0 ; n < number_of_atoms; n++) checkCuda ( cudaFree(d_distance[n]),__LINE__);
    
    checkCuda ( cudaFree(d_distance),__LINE__);
    checkCuda (cudaFree(sdistance), __LINE__);
    
    checkCuda (cudaFree(d_atoms), __LINE__);
    
    //double ret  = *res;
    //cudaFree(res);
    
    return res;
}

