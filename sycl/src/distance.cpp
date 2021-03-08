#include <iostream>

#include<CL/sycl.hpp>

#include<math.h>

#include "distance.h"


using namespace std;

double* distance(vector<atom_st> atoms, int number_of_atoms, int num_of_block){

    
    int size_of_atoms = number_of_atoms*sizeof(atom_st);
    // TODO: distance is useful only on the device, so initiate only in device memory as d_distance
    double* distance = (double*)malloc(2*num_of_block*number_of_atoms*number_of_atoms*sizeof(double));
    atom_st* atoms_tmp = (atom_st*)malloc(num_of_block*size_of_atoms);
    double* res = (double*)malloc(num_of_block*sizeof(double));
    atom_st* d_atoms;
    
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

    //initialize atoms_tmp
    for (int i = 0; i < number_of_atoms * num_of_block; i++){
        atoms_tmp[i] = atoms[i];
    }

    cl::sycl::device gpu = cl::sycl::gpu_selector{}.select_device();

    cl::sycl::queue queue_gpu(gpu, exception_handler);
    
    //SYCL buffer scope
    {
        cl::sycl::buffer<atom_st*, 1> buf_atoms_tmp(&atoms_tmp, cl::sycl::range<1>(1));
        cl::sycl::buffer<double*, 1> buf_distance(&distance, cl::sycl::range<1>(1));
        cl::sycl::buffer<double*, 1> buf_res(&res, cl::sycl::range<1>(1));
        //Computation of the point distance. Compute the Eucledian distance of
        //each atom from each others in the same rotation.
        queue_gpu.submit([&] (cl::sycl::handler& cgh) {

            // Set up device access to data in buffers
            auto d_atoms = buf_atoms_tmp.get_access<cl::sycl::access::mode::read>(cgh);
            auto d_distance = buf_distance.get_access<cl::sycl::access::mode::read>(cgh);
            // auto d_res = buf_res.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            //set up work group size
            cl::sycl::range<1> num_groups{num_of_block};
            cl::sycl::range<1> group_size{number_of_atoms};
            cgh.parallel_for_work_group(num_groups, group_size, [=] (cl::sycl::group<1> grp) {
                
                int bid = grp.get_id(0);
                grp.parallel_for_work_item([&] (cl::sycl::h_item<1> it){
                    
                    int tid = it.get_local_id(0) + bid*number_of_atoms;
                    *d_distance[tid] = 0;
                    double dx,dy,dz,distSqr;
                    float prova = 4.0;
                    for(int j = bid*number_of_atoms; j < number_of_atoms*(bid+1); j++){
                        dx = d_atoms[tid]->position[0] - d_atoms[j]->position[0];
                        dy = d_atoms[tid]->position[1] - d_atoms[j]->position[1];
                        dz = d_atoms[tid]->position[2] - d_atoms[j]->position[2];
                        distSqr = (dx * dx + dy * dy + dz * dz);
                        *d_distance[tid] += cl::sycl::sqrt(distSqr);
                    }
                });
            });

        });
        try{
            queue_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "Get exception in SYCL Eucledian distance computation:\n" 
            << e.what() << std::endl; 
        }

        //Start reduction
        queue_gpu.submit([&](cl::sycl::handler &cgh) {
            // Set up device access to data in buffers
            auto d_atoms = buf_atoms_tmp.get_access<cl::sycl::access::mode::read>(cgh);
            auto d_distance = buf_distance.get_access<cl::sycl::access::mode::read>(cgh);
            auto d_res = buf_res.get_access<cl::sycl::access::mode::read_write>(cgh);
            //set up work group size
            cl::sycl::range<1> num_groups{num_of_block};
            cl::sycl::range<1> group_size{number_of_atoms};
            cgh.parallel_for_work_group(num_groups, group_size, [=] (cl::sycl::group<1> grp) {
                int bid = grp.get_id(0);
                double tmp[number_of_atoms];
                grp.parallel_for_work_item([&] (cl::sycl::h_item<1> it){
                    uint tid = it.get_local_id(0);
                    uint i = tid + bid*number_of_atoms;
                    if(i < (bid + 1)*number_of_atoms){
                        tmp[tid] = *d_distance[i];
                    }
                    else tmp[tid] = 0;
                });// auto sync after this
                
                for(unsigned int s = bid/2; s > 0; s >>= 1){
                    grp.parallel_for_work_item([&] (cl::sycl::h_item<1> it){
                        uint tid = it.get_local_id(0);
                        if(tid < s){
                            tmp[tid] += tmp[tid+s];
                        }
                    });
                }
                //It may not work, problem with sync and save the data
                // should get the thread.id, but still don't know how.
                res[bid] = tmp[0];

            });
        });

        try{
            queue_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "Caught synchronous error in SYCL\n"
            << e.what() << std::endl;
        }
    }; // close the buffer scope and buffer distruction

    free(atoms_tmp);
    free(distance);
    return res;
}