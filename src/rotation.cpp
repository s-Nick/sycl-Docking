#include "rotation.cuh"
#include "cublas_v2.h"
#include "cublas_api.h"
#include "cuda_runtime.h"
#include "atom.cuh"
#include<iostream>


using namespace std;

Rotation::Rotation(float3 vector){
    quaternion = make_float4(vector.x,vector.y,vector.z,0);
}

__host__ void Rotation::getQuaternion(float4* h_quat,float4* unit_quaternion){
    cudaMemcpy(h_quat,unit_quaternion,sizeof(float4),cudaMemcpyDeviceToHost);
}

__device__ void setUnitQuaternion(float* quaternion,float& angle,float4& unit_quaternion, float* norm ){ 

    float sin_2, cos_2;
    //__sincosf(angle/2, &sin_2, &cos_2);

    cos_2 = cosf(angle/2);
    sin_2 = sinf(angle/2);

    quaternion[0] = quaternion[0]/ *norm;
    quaternion[1] = quaternion[1]/ *norm;
    quaternion[2] = (quaternion[2]/ *norm);
    
    unit_quaternion = make_float4(sin_2*(quaternion[0]),sin_2*(quaternion[1]),sin_2*(quaternion[2]),cos_2);

}


__device__ void initFirstRow(int tid, float* rotation_matrix, size_t& pitch, float4 & unit_quat){
    
    float* row_start = (float*)((char*)rotation_matrix  + tid*pitch);
    /*
    row_start[0] =  1-2*(unit_quat[1]*unit_quat[1] + powf(unit_quat[2],2));
    //row_start++;
    row_start[1] = 2*(unit_quat[0]*unit_quat[1] - unit_quat[2]*unit_quat[3]);
    //row_start++;
    row_start[2] = 2*(unit_quat[0]*unit_quat[2] + unit_quat[1]*unit_quat[3]);
*/
    row_start[0] =  1-2*(powf(unit_quat.y,2) + powf(unit_quat.z,2));

    row_start[1] = 2*(unit_quat.x*unit_quat.y - unit_quat.z*unit_quat.w);

    row_start[2] = 2*(unit_quat.x*unit_quat.z + unit_quat.y*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }

}

__device__ void initSecondRow(int tid, float* rotation_matrix, size_t& pitch, float4 & unit_quat){
    
    float* row_start = (float*)((char*)rotation_matrix  + tid*pitch);
    row_start[0] =  2*(unit_quat.x*unit_quat.y + unit_quat.z*unit_quat.w);

    row_start[1] = 1-2*(unit_quat.x*unit_quat.x + powf(unit_quat.z,2));//unit_quat.z*unit_quat.z);

    row_start[2] = 2*(unit_quat.y*unit_quat.z - unit_quat.x*unit_quat.w);
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }
}

__device__ void initThirdRow(int tid, float* rotation_matrix, size_t & pitch, float4 & unit_quat){
    
    float* row_start = (float*)((char*)rotation_matrix  + tid*pitch);
    row_start[0] =  2*(unit_quat.x*unit_quat.z - unit_quat.y*unit_quat.w);

    row_start[1] = 2*(unit_quat.y*unit_quat.z + unit_quat.x*unit_quat.w);

    row_start[2] = 1-2*(powf(unit_quat.x,2) + powf(unit_quat.y,2));
    for(int i= 0;i < 3;i++){
        if(row_start[i] == -0) row_start[i] = 0;
    }
}

__global__ void init_rot_matrix_kernel(float* quat,float angle,
                                        float* rotation_matrix, size_t pitch, float* norm){
    
    float4 unit_quaternion;
    

    setUnitQuaternion(quat,angle,unit_quaternion,norm);
    
    //initFirstRow(0,rotation_matrix,pitch,unit_quaternion);
    //initSecondRow(1,rotation_matrix,pitch,unit_quaternion);
    //initThirdRow(2,rotation_matrix,pitch,unit_quaternion);
    
    
    
    int tid = threadIdx.x; //+ blockIdx.x*blockDim.x;
    if(tid < 3){
        if(tid == 0) initFirstRow(tid,rotation_matrix,pitch,unit_quaternion);
        else if(tid == 1) initSecondRow(tid,rotation_matrix,pitch,unit_quaternion);
        else if(tid == 2) initThirdRow(tid,rotation_matrix,pitch,unit_quaternion);
    }
}

__global__ void rotation_kernel(atom_st* res, atom_st* atoms, float* rotation_matrix, size_t pitch,
                                int number_of_atoms, float3 pp){
    
    
    float* first_row = (float*)((char*)rotation_matrix  + 0*pitch);
    float* second_row = (float*)((char*)rotation_matrix + 1*pitch);
    float* third_row = (float*)((char*)rotation_matrix  + 2*pitch);

    int tid = threadIdx.x;
    if(tid < number_of_atoms){
        
        res[tid].position.x = atoms[tid].position.x * first_row[0] + \
                              atoms[tid].position.y * first_row[1] + \
                              atoms[tid].position.z * first_row[2] + pp.x;
        
        res[tid].position.y = atoms[tid].position.x * second_row[0] + \
                              atoms[tid].position.y * second_row[1] + \
                              atoms[tid].position.z * second_row[2] + pp.y;
        
        res[tid].position.z = atoms[tid].position.x * third_row[0] + \
                              atoms[tid].position.y * third_row[1] + \
                              atoms[tid].position.z * third_row[2] + pp.z;
    }

}

__global__ void first_translation(atom_st* atoms,float3 pp,int number_of_atoms){

    int tidx = threadIdx.x;
     
    if(tidx < number_of_atoms){
        atoms[tidx].position.x -= pp.x;
        atoms[tidx].position.y -= pp.y;
        atoms[tidx].position.z -= pp.z;
    }

}

__global__ void back_translation(atom_st* atoms,float3 pp,int number_of_atoms){

    int tidx = threadIdx.x;
     
    if(tidx < number_of_atoms){
        atoms[tidx].position.x += pp.x;
        atoms[tidx].position.y += pp.y;
        atoms[tidx].position.z += pp.z;
    }

}

void Rotation::normalize(float* quat, float* norm){
    cublasHandle_t handle;
    cublasStatus_t stat;
    
    stat = cublasCreate(&handle);

    stat = cublasSnrm2(handle,4,quat,1,norm);
    
    cublasDestroy(handle);

}

atom_st* Rotation::rotate(float angle, std::vector<atom_st>& atoms_st, float3& pp){

    int deviceId;
    float *quat;
    atom_st *atoms;
    float* rotation_matrix;
    size_t pitch;
    atom_st * res;

    cudaError_t err;

    int number_of_atoms = atoms_st.size();
    int size_of_atoms = number_of_atoms*sizeof(atom_st);

    cudaMallocManaged(&quat, 4*sizeof(float));
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
    
    cudaGetDevice(&deviceId);
    //cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    cudaMemPrefetchAsync(quat,4*sizeof(float),deviceId);
    cudaMemPrefetchAsync(atoms,size_of_atoms,deviceId);
    cudaMemPrefetchAsync(res,size_of_atoms,deviceId);

    cudaMallocPitch(&rotation_matrix, &pitch, 3*sizeof(float), 3);

    float4* unit_quaternion;
    cudaMallocManaged(&unit_quaternion, sizeof(float4));

    cout << atoms[0].position.x << " " << atoms[0].position.y << " " << atoms[0].position.z << endl;
    float3 passingPoint = pp;
    
    
    first_translation<<<1,number_of_atoms>>>(atoms,passingPoint, number_of_atoms);

    cudaDeviceSynchronize();
    cout << "traslation\n";
    cout << atoms[0].position.x << " " << atoms[0].position.y << " " << atoms[0].position.z << endl;

    float norm;
    
    normalize(quat,&norm);
    
    float* g_norm;
    cudaMalloc(&g_norm, sizeof(float));
    cudaMemcpy(g_norm, &norm, sizeof(float), cudaMemcpyHostToDevice);
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

    cout << "norm of quat: " << norm << endl;

    cout << quaternion.x << " " << quaternion.y << " " << quaternion.z << endl;

    /*err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Error %s \n", cudaGetErrorString(err));
    }*/

    float* h_rot_matrix = (float*)malloc(3*3 *sizeof(float));
    cout << "first atom : " << __LINE__ << endl;
    cout << res[0].position.x << " " << res[0].position.y << " " << res[0].position.z << endl;
    /*
    for(int i = 0; i< 3;i++){
        h_rot_matrix[i] = (float*)malloc(3*sizeof(float));
    }

    for(int i = 0; i< 3;i++){
        for(int j = 0;j<3;j++){
            h_rot_matrix[i][j] = i+j; 
        }
    }
*/
    size_t h_pitch = pitch;
    cudaMemcpy2D(h_rot_matrix,3*sizeof(float),rotation_matrix,pitch,3*(sizeof(float)),3, cudaMemcpyDeviceToHost);

    for(int i = 0; i< 3; i++){
        for(int j = 0; j<3;j++){
            cout << h_rot_matrix[i*3+j] << " ";
        }
        cout << endl;
    }

    //cout << h_rot_matrix << endl;

    for(int i = 0; i< 4;i++) cout << quat[i] << " ";
    cout << endl;
    cout << "here\n";

    //float4 * h_uq;

    //getQuaternion(h_uq,unit_quaternion);

    //cudaFree(h_uq);
    //cudaFree(res);
    cudaFree(quat);
    cudaFree(atoms);
    cudaFree(unit_quaternion);
    cudaFree(g_norm);

    free(h_rot_matrix);

    return res;
}