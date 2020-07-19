#include "rotation.cuh"
#include "cublas_v2.h"
#include "cublas_api.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include <stdio.h>
#include "atom.cuh"
#include <iostream>


using namespace std;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}


Rotation::Rotation(double3 vector){
    quaternion = make_double4(vector.x,vector.y,vector.z,0);
}

__host__ void Rotation::getQuaternion(double4* h_quat,double4* unit_quaternion){
    cudaMemcpy(h_quat,unit_quaternion,sizeof(double4),cudaMemcpyDeviceToHost);
}

__device__ void setUnitQuaternion(double* quaternion,double angle,double4& unit_quaternion, double* norm ){ 

    double sin_2, cos_2;
    //__sincosf(angle/2, &sin_2, &cos_2);
    cos_2 = cos(angle/2);
    sin_2 = sin(angle/2);


    quaternion[0] = quaternion[0]/ *norm;
    quaternion[1] = quaternion[1]/ *norm;
    quaternion[2] = (quaternion[2]/ *norm);
    
    unit_quaternion = make_double4(sin_2*(quaternion[0]),sin_2*(quaternion[1]),sin_2*(quaternion[2]),cos_2);

}


__device__ void initFirstRow(int tid, double* rotation_matrix, size_t& pitch, double4 & unit_quat){
    
    double* row_start = (double*)((char*)rotation_matrix  + tid*pitch);
    /*
    row_start[0] =  1-2*(unit_quat[1]*unit_quat[1] + powf(unit_quat[2],2));
    //row_start++;
    row_start[1] = 2*(unit_quat[0]*unit_quat[1] - unit_quat[2]*unit_quat[3]);
    //row_start++;
    row_start[2] = 2*(unit_quat[0]*unit_quat[2] + unit_quat[1]*unit_quat[3]);
*/  
    
    row_start[0] =  1-2*(pow(unit_quat.y,2) + pow(unit_quat.z,2));
    
    row_start[1] = 2*(unit_quat.x*unit_quat.y - unit_quat.z*unit_quat.w);

    row_start[2] = 2*(unit_quat.x*unit_quat.z + unit_quat.y*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }
    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);


}

__device__ void initSecondRow(int tid, double* rotation_matrix, size_t& pitch, double4 & unit_quat){
    
    double* row_start = (double*)((char*)rotation_matrix  + tid*pitch);
    row_start[0] =  2*(unit_quat.x*unit_quat.y + unit_quat.z*unit_quat.w);

    row_start[1] = 1-2*(unit_quat.x*unit_quat.x + pow(unit_quat.z,2));//unit_quat.z*unit_quat.z);

    row_start[2] = 2*(unit_quat.y*unit_quat.z - unit_quat.x*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }

    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);

}

__device__ void initThirdRow(int tid, double* rotation_matrix, size_t & pitch, double4 & unit_quat){
    
    double* row_start = (double*)((char*)rotation_matrix  + tid*pitch);
    row_start[0] =  2*(unit_quat.x*unit_quat.z - unit_quat.y*unit_quat.w);

    row_start[1] = 2*(unit_quat.y*unit_quat.z + unit_quat.x*unit_quat.w);

    row_start[2] = 1-2*(pow(unit_quat.x,2) + pow(unit_quat.y,2));
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }
    //printf("row %d values : %lf %lf %lf\n", tid,row_start[0],row_start[1],row_start[2]);

}

__global__ void init_rot_matrix_kernel(double* quat,double angle,
                                        double* rotation_matrix, size_t pitch, double* norm){
    
    double4 unit_quaternion;
    

    setUnitQuaternion(quat,angle,unit_quaternion,norm);
    
    //initFirstRow(0,rotation_matrix,pitch,unit_quaternion);
    //initSecondRow(1,rotation_matrix,pitch,unit_quaternion);
    //initThirdRow(2,rotation_matrix,pitch,unit_quaternion);
    
    //printf("unit_q %lf x %lf y %lf z %lf w\n", unit_quaternion.x,unit_quaternion.y,unit_quaternion.z,unit_quaternion.w);

    int tid = threadIdx.x; //+ blockIdx.x*blockDim.x;
    if(tid < 3){
        if(tid == 0) initFirstRow(tid,rotation_matrix,pitch,unit_quaternion);
        else if(tid == 1) initSecondRow(tid,rotation_matrix,pitch,unit_quaternion);
        else if(tid == 2) initThirdRow(tid,rotation_matrix,pitch,unit_quaternion);
    }
}

__global__ void rotation_kernel(atom_st* res, atom_st* atoms, double* rotation_matrix, size_t pitch,
                                int number_of_atoms, double3 pp){
    
    
    double* first_row = (double*)((char*)rotation_matrix  + 0*pitch);
    double* second_row = (double*)((char*)rotation_matrix + 1*pitch);
    double* third_row = (double*)((char*)rotation_matrix  + 2*pitch);

    int tid = threadIdx.x;
    if(tid < number_of_atoms){
        
        res[tid].id = atoms[tid].id;

        res[tid].position.x = atoms[tid].position.x * first_row[0] + \
                              atoms[tid].position.y * first_row[1] + \
                              atoms[tid].position.z * first_row[2] + pp.x;
        
        res[tid].position.y = atoms[tid].position.x * second_row[0] + \
                              atoms[tid].position.y * second_row[1] + \
                              atoms[tid].position.z * second_row[2] + pp.y;
        
        res[tid].position.z = atoms[tid].position.x * third_row[0] + \
                              atoms[tid].position.y * third_row[1] + \
                              atoms[tid].position.z * third_row[2] + pp.z;
    
    //printf("%d %lf x %lf y %lf z\n", tid,res[tid].position.x,res[tid].position.y,res[tid].position.z);
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

void Rotation::normalize(double* quat, double* norm){
    cublasHandle_t handle;
    cublasStatus_t stat;
    
    stat = cublasCreate(&handle);

    stat = cublasDnrm2(handle,4,quat,1,norm);
    
    cublasDestroy(handle);

}

atom_st* Rotation::rotate(double angle, std::vector<atom_st>& atoms_st, double3& pp){

    int deviceId;
    double *quat;
    atom_st *atoms;
    double* rotation_matrix;
    size_t pitch;
    atom_st * res;

    cudaError_t err;

    int number_of_atoms = atoms_st.size();
    int size_of_atoms = number_of_atoms*sizeof(atom_st);

    checkCuda( cudaGetDevice(&deviceId) );
    checkCuda( cudaMallocManaged(&quat, 4*sizeof(double)));
    cudaMallocManaged(&atoms, size_of_atoms);
    cudaMallocManaged(&res, size_of_atoms);
    //initialize vector of quaternion
    quat[0] = quaternion.x;
    quat[1] = quaternion.y;
    quat[2] = quaternion.z;
    quat[3] = quaternion.w;
    
    //initialize vector of atoms
    int i = 0;
    for(auto at : atoms_st){
        atoms[i] = at;
        i++;
    }
    
    
    //cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    cudaMemPrefetchAsync(quat,4*sizeof(double),deviceId);
    cudaMemPrefetchAsync(atoms,size_of_atoms,deviceId);
    cudaMemPrefetchAsync(res,size_of_atoms,deviceId);

    cudaMallocPitch(&rotation_matrix, &pitch, 3*sizeof(double), 3);

    //double4* unit_quaternion;
    //cudaMallocManaged(&unit_quaternion, sizeof(double4));

    //cout << atoms[0].position.x << " " << atoms[0].position.y << " " << atoms[0].position.z << endl;
    double3 passingPoint = pp;
    
    
    first_translation<<<1,number_of_atoms>>>(atoms,passingPoint, number_of_atoms);

    cudaDeviceSynchronize();
    //cout << "traslation\n";
    //cout << atoms[0].position.x << " " << atoms[0].position.y << " " << atoms[0].position.z << endl;

    double norm;
    
    normalize(quat,&norm);
    
    double* g_norm;
    cudaMalloc(&g_norm, sizeof(double));
    cudaMemcpy(g_norm, &norm, sizeof(double), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }
    

    init_rot_matrix_kernel<<<1,3>>>(quat, angle, rotation_matrix, pitch, g_norm);
    cudaDeviceSynchronize();
    rotation_kernel<<<1,number_of_atoms>>>(res,atoms,rotation_matrix,pitch,number_of_atoms,passingPoint); 
    
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    back_translation<<<1,number_of_atoms>>>(atoms,passingPoint,number_of_atoms);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        cout << __LINE__ << endl;
        printf("Error %s \n", cudaGetErrorString(err));
    }

    //cout << "norm of quat: " << norm << endl;

    //cout << quaternion.x << " " << quaternion.y << " " << quaternion.z << endl;

    /*err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }*/
    
    /*float* h_rot_matrix = (float*)malloc(3*3 *sizeof(float));
    /*cout << "first atom : " << __LINE__ << endl;
    for(int i = 0; i < number_of_atoms; i++){
        cout << res[i].id << " " << res[i].position.x << " " << res[i].position.y << " " << res[i].position.z << endl;
    }*/
    /*
    for(int i = 0; i< 3;i++){
        h_rot_matrix[i] = (float *)malloc(3*sizeof(float));
    }

    for(int i = 0; i< 3;i++){
        for(int j = 0;j<3;j++){
            h_rot_matrix[i][j] = i+j; 
        }
    }

    size_t h_pitch = pitch;
    cudaMemcpy2D(h_rot_matrix,3*sizeof(float),rotation_matrix,pitch,3*(sizeof(float)),3, cudaMemcpyDeviceToHost);

    for(int i = 0; i< 3; i++){
        for(int j = 0; j<3;j++){
            cout << h_rot_matrix[i*3+j] << " ";
        }
        cout << endl;
    }
    
    //cout << h_rot_matrix << endl;
    /*
    for(int i = 0; i< 4;i++) cout << quat[i] << " ";
    cout << endl;
    cout << "here\n";
    */
    //float4 * h_uq;

    //getQuaternion(h_uq,unit_quaternion);

    //cudaFree(h_uq);
    //cudaFree(res);
    cudaFree(quat);
    cudaFree(atoms);
    //cudaFree(unit_quaternion);
    cudaFree(g_norm);
    checkCuda( cudaFree(rotation_matrix));
    //free(h_rot_matrix);

    return res;
}