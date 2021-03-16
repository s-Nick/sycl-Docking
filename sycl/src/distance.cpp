#include <iostream>

#include<CL/sycl.hpp>

#include<math.h>

#include "distance.h"

#define NUM_OF_BLOCKS 360

using namespace std;

class distance;

double* distance(vector<atom_st> atoms, int number_of_atoms, int num_of_block){

    //SYCL error handler similar to checkCuda, but lambda fun
    auto exception_handler = [](cl::sycl::exception_list exceptions)
    {
        for (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const &e)
            {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    };
    
    cl::sycl::device gpu = cl::sycl::gpu_selector{}.select_device();
    cl::sycl::queue queue_gpu(gpu, exception_handler, cl::sycl::property::queue::in_order());
    
    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    // TODO: distance is useful only on the device, so initiate only in device memory as d_distance
    double* d_distance = 
        cl::sycl::malloc_device<double>(2*num_of_block*number_of_atoms*number_of_atoms*sizeof(double), queue_gpu);
    
    atom_st* d_atoms = cl::sycl::malloc_shared<atom_st>(num_of_block*size_of_atoms, queue_gpu);

    //Since use buffer and accessor give me memory problem, switch to manual memory handling
    double* d_res = cl::sycl::malloc_shared<double>(num_of_block*sizeof(double),queue_gpu);
    //atom_st* d_atoms;
    
    double* res = (double*)std::malloc(num_of_block*sizeof(double));
        
    
    //initialize atoms_tmp
    for (int i = 0; i < number_of_atoms * num_of_block; i++){
        d_atoms[i] = atoms[i];
    }

    //std::cout << "DISTANCE.cpp line " << __LINE__ << std::endl;

    //SYCL buffer scope
    {
        
        queue_gpu.submit([&](cl::sycl::handler &cgh){
            
            // NEEDED FOR DEBUGGING IT BREAKS EVERYTHING
            //cl::sycl::stream out(8184, 256, cgh);

            cl::sycl::range<1> total{512*NUM_OF_BLOCKS};
            cl::sycl::range<1> group{512};

            //Allocate buffer for reduction holder;
            cl::sycl::range<1> tmp_range(1024);

            cl::sycl::accessor<double, 1, cl::sycl::access_mode::discard_read_write,
                                cl::sycl::access::target::local>
                tmp_acc(tmp_range, cgh ); 

            cgh.parallel_for<class distance>(cl::sycl::nd_range<1>(total, group),
                            [=](cl::sycl::nd_item<1> it){

                int bid = it.get_group().get_id(0);
                
                //double *tmp = (double *)malloc(number_of_atoms * sizeof(double));
                //double tmp[1024];
                int tid = it.get_local_id(0) + bid*number_of_atoms;
                d_distance[tid] = 0;
                if(tid < number_of_atoms*(bid+1)){
                    double dx, dy, dz, distSqr;
                    for(int j = number_of_atoms*bid; j < number_of_atoms*(bid + 1 ); j++){
                        dx = d_atoms[tid].position.x() - d_atoms[j].position.x();
                        dy = d_atoms[tid].position.y() - d_atoms[j].position.y();
                        dz = d_atoms[tid].position.z() - d_atoms[j].position.z();
                        distSqr = (dx * dx + dy * dy + dz * dz);
                        d_distance[tid] += cl::sycl::sqrt<double>(distSqr);
                    }
                }

                it.barrier();

                unsigned int tidx = it.get_local_id(0);
                unsigned int index = tidx + bid*number_of_atoms;

                if( index < (bid+1)* number_of_atoms ){
                    tmp_acc[tidx] = d_distance[index];
                } 
                else
                    tmp_acc[tidx] = 0;

                it.barrier();
                

                
                for(unsigned int s = it.get_local_range(0)/2 ; s > 0; s >>= 1 ){
                    if( tidx < s ){
                        tmp_acc[tidx] += tmp_acc[tidx+s];
                    }
                    it.barrier();
                } 

                it.barrier();
                if(tidx == 0){
                    d_res[bid] = tmp_acc[tidx];
                    //out << "reduction: " << d_res[bid] << " " << bid << cl::sycl::endl;
                }
                
            
            });

        });
        


        /* OLD VERSION OF DISTANCE COMPUTATION THAT USE BUFFERS AND ACCESSORS.
        
        //cl::sycl::buffer<atom_st*> atoms_tmp(size_of_atoms);
        //cl::sycl::buffer<double*> distance(size_of_atoms*num_of_block);
        //cl::sycl::buffer<double*> res(sizeof(double)*num_of_block*360);
        //cl::sycl::buffer<double*> buf_res(&res, cl::sycl::range<1>());
        //Computation of the point distance. Compute the Eucledian distance of
        //each atom from each others in the same rotation.
        queue_gpu.submit([&] (cl::sycl::handler& cgh) {
            

            // NEEDED FOR DEBUGGING IT BREAKS EVERYTHING
            cl::sycl::stream out(2048, 256, h);

            // Set up device access to data in buffers
            //auto d_atoms = atoms_tmp.get_access<cl::sycl::access::mode::read>(cgh);
            //auto d_distance = distance.get_access<cl::sycl::access::mode::read>(cgh);
            //auto d_res = res.get_access<cl::sycl::access::mode::read_write>(cgh);

            //set up work group size
            cl::sycl::range<1> num_groups{ 360 }; //in case it works for the atoms, use sizeof
            
            // Using 64 create problem try to fix this here
            cl::sycl::range<1> group_size{ 64 }; //should be number_of_atoms 
            cgh.parallel_for_work_group<class distance>(num_groups, group_size, [=] (cl::sycl::group<1> grp) {
                
                int bid = grp.get_id(0);

                //if(bid < number_of_atoms){
                    double *tmp = (double *)malloc(number_of_atoms * sizeof(double));

                    grp.parallel_for_work_item([&] (cl::sycl::h_item<1> it){
                        
                        int tid = it.get_local() + bid*number_of_atoms;
                        d_distance[tid] = 0;
                        if(tid < number_of_atoms*(bid+1)){
                            double dx,dy,dz,distSqr;
                            //float prova = 4.0;
                            for(int j = bid*number_of_atoms; j < number_of_atoms*(bid+1); j++){
                                dx = d_atoms[tid].position[0] - d_atoms[j].position[0];
                                dy = d_atoms[tid].position[1] - d_atoms[j].position[1];
                                dz = d_atoms[tid].position[2] - d_atoms[j].position[2];
                                distSqr = (dx * dx + dy * dy + dz * dz);
                                d_distance[tid] += cl::sycl::sqrt(distSqr);
                            }
                        }
                    });
                    
                    grp.mem_fence();
                    //int bid = grp.get_id(0);
                    //double *tmp = (double *)malloc(number_of_atoms * sizeof(double));
                    grp.parallel_for_work_item([&](cl::sycl::h_item<1> it){
                            uint tid = it.get_local();
                            uint i = tid + bid * number_of_atoms;
                            if (i < (bid + 1) * number_of_atoms){
                                tmp[tid] = d_distance[i];
                            }
                            else
                                tmp[tid] = 0;
                    }); // auto sync after this
                    grp.mem_fence();
                    
                    for (unsigned int s = bid / 2; s > 0; s >>= 1){
                        grp.parallel_for_work_item([&](cl::sycl::h_item<1> it){
                            uint tid = it.get_local();
                            if (tid < s){
                                tmp[tid] += tmp[tid + s];
                            }
                        });
                        grp.mem_fence();
                    }

                    //It may not work, problem with sync and save the data
                    // should get the thread.id, but still don't know how.
                    
                    grp.mem_fence();
                    
                    d_res[bid] = tmp[0];
                    
                //}
            });
            
        });
        */    


        try{
            queue_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "Get exception in SYCL Eucledian distance computation line " << __LINE__ << ": \n" 
            << e.what() << std::endl; 
        }

        

        //std::cout << "DISTANCE.cpp line " << __LINE__ << std::endl;
        //queue_gpu.memcpy(res,d_res,num_of_block*sizeof(double));
    }; // close the buffer scope and buffer distruction
    
    //std::cout << "DISTANCE.cpp line " << __LINE__ << std::endl;

    for(int i = 0; i < num_of_block; i++){
        res[i] = d_res[i];
    }

    //DEBUG PRINTING
    /*
    for(int i = 0 ; i < num_of_block; i++){
        std::cout << "angle : " << i <<  " total distance: ";
        std::cout << res[i] <<  std::endl;
    }
    */
    cl::sycl::free(d_res, queue_gpu);
    cl::sycl::free(d_atoms,queue_gpu);
    cl::sycl::free(d_distance, queue_gpu);
    
    return res;
}