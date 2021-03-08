
//#include<CL/sycl.hpp>

#include "rotation.h"

#include "math_constants.h"
#include<stdio.h>
#include<iostream>

#include<CL/sycl.hpp>

#define NUM_OF_BLOCKS 360
class rotation;
/**
 * This function is used to set the mememory of the host and the device in order to compute the rotation using the rotation kernel.
 * At the end of the computation all the rotated positions are brought to the device memory and stored in a vector of vectors for 
 * future usage.
 * 
 * @param angle The angle of the first rotation of the block.
 * @param atoms_st Vector containing all the atoms to rotate.
 * @param pp PassingPoint, point belonging to the axis of the rotation, used to compute the rotation.
 * @param unit_quaternion The vector containing all the computed unit_quaternions, one for each rotation.
 **/
std::vector<std::vector<atom_st>> Rotation::rotate(int angle, std::vector<atom_st> &atoms_st,
                                                   cl::sycl::double3 &pp, 
                                                   cl::sycl::double4 *unit_quaternion){

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

    int number_of_atoms = atoms_st.size();
    int size_of_atoms = number_of_atoms*sizeof(atoms_st);

    atom_st* atoms = (atom_st*)malloc(size_of_atoms);

    

    cl::sycl::device gpu =cl::sycl::gpu_selector{}.select_device();
    cl::sycl::queue q_gpu(gpu, exception_handler);

    atom_st *h_res = (atom_st *)cl::sycl::malloc_host(size_of_atoms * NUM_OF_BLOCKS, q_gpu);
    atom_st *d_res = (atom_st *)cl::sycl::malloc_device(size_of_atoms * NUM_OF_BLOCKS, q_gpu);

    //initialize vector of atoms
    int i = 0;
    for (auto at : atoms_st){
        atoms[i] = at;
        i++;
    }
    //cl::sycl::double3 passingPoint = pp;
    {//SYCL buffer scope

        cl::sycl::buffer<atom_st*, 1 > buf_atoms(&atoms,cl::sycl::range<1>(1));
        cl::sycl::buffer<cl::sycl::double3,1> buf_passingPoint(&pp, cl::sycl::range<1>(1));
        cl::sycl::buffer<cl::sycl::double4,1> buf_unitQuaternion(unit_quaternion, cl::sycl::range<1>(1));
        
        q_gpu.submit([&] (cl::sycl::handler& h) {

            auto d_atoms = buf_atoms.get_access<cl::sycl::access::mode::read_write>(h);
            auto d_pp = buf_passingPoint.get_access<cl::sycl::access::mode::read>(h);
            auto d_unitQuaternion = buf_unitQuaternion.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class rotation>(cl::sycl::range<1>(number_of_atoms), [=](cl::sycl::id<1> tidx)  {
                    //the right part may be wrong in the way it gets the number
                    d_atoms[tidx]->position[0] -= d_pp[0][0];
                    d_atoms[tidx]->position[1] -= d_pp[0][1];
                    d_atoms[tidx]->position[2] -= d_pp[0][2];
            });

            cl::sycl::range<1> num_groups{NUM_OF_BLOCKS};
            cl::sycl::range<1> group_size{number_of_atoms};
            h.parallel_for_work_group(num_groups, group_size, [=](cl::sycl::group<1> grp){
                int bid = grp.get_id(0);
                double rot_matrix[3][3];
                // Initialize rotation matrix in 
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> it){
                    // Initialize First row of the rotation matrix
                    if( it.get_local_id() == 0 ){
                        rot_matrix[0][0] = 1-2*(cl::sycl::pow<double>(d_unitQuaternion[bid][1],2)) + \
                                            cl::sycl::pow<double>(d_unitQuaternion[bid][2],2);
                        rot_matrix[0][1] = 2 * (d_unitQuaternion[bid][0] * d_unitQuaternion[bid][1] - \
                                                d_unitQuaternion[bid][2] * d_unitQuaternion[bid][3]);
                        rot_matrix[0][2] = 2 * (d_unitQuaternion[bid][0] * d_unitQuaternion[bid][3] + \
                                                d_unitQuaternion[bid][1] * d_unitQuaternion[bid][3]);
                    }//be careful on the sign of 0, it may cause error in computation.
                    else if(it.get_local_id() == 1){
                        rot_matrix[1][0] = 2 * (d_unitQuaternion[bid][0] * d_unitQuaternion[bid][1] + \
                                                d_unitQuaternion[bid][2] * d_unitQuaternion[bid][3]);
                        rot_matrix[1][1] = 1 - 2 * (cl::sycl::pow<double>(d_unitQuaternion[bid][0], 2) +
                                                    cl::sycl::pow<double>(d_unitQuaternion[bid][2], 2));
                        rot_matrix[1][2] = 2 * (d_unitQuaternion[bid][1] * d_unitQuaternion[bid][2] - \
                                                d_unitQuaternion[bid][0] * d_unitQuaternion[bid][3]);
                    }
                    else if( it.get_local_id() == 2){
                        rot_matrix[2][0] = 2 * (d_unitQuaternion[bid][0] * d_unitQuaternion[bid][2] - \
                                                d_unitQuaternion[bid][1] * d_unitQuaternion[bid][3]);
                        rot_matrix[2][1] = 2 * (d_unitQuaternion[bid][1] * d_unitQuaternion[bid][2] + \
                                                d_unitQuaternion[bid][0] * d_unitQuaternion[bid][3]);
                        rot_matrix[2][2] = 1 - 2 * (cl::sycl::pow<double>(d_unitQuaternion[bid][0], 2) + \
                                                    cl::sycl::pow<double>(d_unitQuaternion[bid][1], 2));
                    }
                });
                grp.parallel_for_work_item([&] (cl::sycl::h_item<1> it){
                    int tidx = it.get_local_id();
                    int index = tidx + bid*number_of_atoms;
                    if(index < number_of_atoms*(bid+1) && number_of_atoms*bid <= index ){
                        
                        d_res[index].id = d_atoms[tidx]->id;

                        d_res[index].position[0] = d_atoms[tidx]->position[0] * rot_matrix[0][0] + \
                                                d_atoms[tidx]->position[1] * rot_matrix[0][1] + \
                                                d_atoms[tidx]->position[2] * rot_matrix[0][2] + d_pp[0][0];

                        d_res[index].position[1] = d_atoms[tidx]->position[0] * rot_matrix[1][0] + \
                                                   d_atoms[tidx]->position[1] * rot_matrix[1][1] + \
                                                   d_atoms[tidx]->position[2] * rot_matrix[1][2] + d_pp[0][1];

                        d_res[index].position[2] = d_atoms[tidx]->position[0] * rot_matrix[2][0] + \
                                                   d_atoms[tidx]->position[1] * rot_matrix[2][1] + \
                                                   d_atoms[tidx]->position[2] * rot_matrix[2][2] + d_pp[0][2];
                    }
                });//second for_work_item end
            });// for_work_group end

        });// submit end
        try{
            q_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "SYCL: synchronous exception occured, Rotation class " << __LINE__ << "\n"
            << e.what() << std::endl; 
        }

        q_gpu.memcpy(h_res, d_res, size_of_atoms * NUM_OF_BLOCKS);
        q_gpu.wait();
    }// SYCL BUFFER Scope

    std::vector<std::vector<atom_st>> result_to_return;
    std::vector<atom_st> tmp;
    //copy the results in order to free the memory and to pass the result to other functions for further usage
    for (int i = 0; i < NUM_OF_BLOCKS; i++)
    {
        for (int c = atoms_st.size() * i; c < atoms_st.size() * (i + 1); c++)
        {
            tmp.push_back(h_res[c]);
        }
        result_to_return.push_back(tmp);
        tmp.clear();
    }

    cl::sycl::free(d_res,q_gpu);
    cl::sycl::free(h_res,q_gpu);
    return result_to_return;
}