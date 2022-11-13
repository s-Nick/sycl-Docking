
//#include<CL/sycl.hpp>

#include "rotation.h"

#include<cmath>
#include<stdio.h>
#include<iostream>

#include<CL/sycl.hpp>

#define NUM_OF_BLOCKS 360
class rotation;
class translation;
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
                                                   cl::sycl::double4 *d_unit_quaternion,
                                                   cl::sycl::queue& q_gpu){

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

    const int number_of_atoms = atoms_st.size();
    int size_of_atoms = number_of_atoms*sizeof(atoms_st);

   
    //cl::sycl::device gpu =cl::sycl::host_selector{}.select_device();
    //cl::sycl::device gpu =cl::sycl::gpu_selector{}.select_device();
    //cl::sycl::queue q_gpu(gpu, exception_handler);

    //std::cout << gpu.get_info<cl::sycl::info::device::vendor>() << std::endl;

     //atom_st* atoms = (atom_st*)malloc(size_of_atoms);

    atom_st* atoms = cl::sycl::malloc_shared<atom_st>(size_of_atoms,q_gpu);


    atom_st *res = cl::sycl::malloc_shared<atom_st>(size_of_atoms * NUM_OF_BLOCKS*number_of_atoms, q_gpu);

    //cl::sycl::double4* d_unit_quaternion = cl::sycl::malloc_shared<cl::sycl::double4>(sizeof(cl::sycl::double4)*NUM_OF_BLOCKS, q_gpu);
   
    

    //initialize vector of atoms
    int i = 0;
    for (auto at : atoms_st){
        atoms[i] = at;
        i++;
    }
    
    
    cl::sycl::double3* passingPoint = cl::sycl::malloc_shared<cl::sycl::double3>(sizeof(cl::sycl::double3),q_gpu);

    passingPoint->x() = pp.x();
    passingPoint->y() = pp.y();
    passingPoint->z() = pp.z();

    q_gpu.wait();
    {//SYCL buffer scope


            q_gpu.parallel_for<class translation>(cl::sycl::nd_range<1>(128,128),
                                             [=](cl::sycl::nd_item<1>it){           
                //int bid = it.get_group().get_id();
                
                //out << "bid number " << bid << cl::sycl::endl;

                int tidx = it.get_local_id();
                
                //the right part may be wrong in the way it gets the number
                if( tidx < number_of_atoms){
                    //out << "threadId.x " <<int(tidx) << cl::sycl::endl;
                    atoms[tidx].position.x() -= passingPoint->x();//[0][0];
                    atoms[tidx].position.y() -= passingPoint->y();//[0][1];
                    atoms[tidx].position.z() -= passingPoint->z();//[0][2];
                }

            });// End tranlsation parallel_for
        //}); // end translation submit
        //q_gpu.wait();
        //std::cout << "rotation line " << __LINE__ << std::endl;
        //translation_s.wait();
        q_gpu.wait_and_throw();
        /*
        try{
            q_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "SYCL: synchronous exception occured, Rotation class " << __LINE__ << "\n"
            << e.what() << std::endl; 
        }
        */
        q_gpu.submit([&] (cl::sycl::handler& h) {
            
            //h.depends_on(translation_s);

            // NEEDED FOR DEBUGGING
            //cl::sycl::stream out(2048, 256, h);

            cl::sycl::range<1> global {NUM_OF_BLOCKS*128};
            cl::sycl::range<1> local {128};

            cl::sycl::range<2> rot_matrix_range(3,3);

            cl::sycl::accessor<double, 2, cl::sycl::access_mode::discard_read_write,
                                cl::sycl::access::target::local>
                rot_matrix_acc(rot_matrix_range, h ); 

            h.parallel_for<class rotation>(cl::sycl::nd_range<1>(global,local), 
                                           [=](cl::sycl::nd_item<1> grp){
                
                int bid = grp.get_group().get_id();

                //double rot_matrix[3][3];

                //Initialize Rotation matrix;
                if(bid < 360){
                    //out << "bid " << bid << cl::sycl::endl;
                    
                    if( grp.get_local_id(0) == 0 ){ //get_local_id
                        
                        rot_matrix_acc[0][0] = 1-2*( d_unit_quaternion[bid].y() * d_unit_quaternion[bid].y()  + \
                                                 d_unit_quaternion[bid].z() * d_unit_quaternion[bid].z());
                        rot_matrix_acc[0][1] = 2 * (d_unit_quaternion[bid].x() * d_unit_quaternion[bid].y() - \
                                                d_unit_quaternion[bid].z() * d_unit_quaternion[bid].w());
                        rot_matrix_acc[0][2] = 2 * (d_unit_quaternion[bid].x() * d_unit_quaternion[bid].z() + \
                                                d_unit_quaternion[bid].y() * d_unit_quaternion[bid].w());
                        //out << "bid " << bid  << " tidx " << grp.get_local_id(0)<< cl::sycl::endl;
                    }//be careful on the sign of 0, it may cause error in computation.
                    else if(grp.get_local_id(0) == 1){
                        rot_matrix_acc[1][0] = 2 * (d_unit_quaternion[bid].x() * d_unit_quaternion[bid].y() + \
                                                d_unit_quaternion[bid].z() * d_unit_quaternion[bid].w());
                        rot_matrix_acc[1][1] = 1 - 2 * ( d_unit_quaternion[bid].x() * d_unit_quaternion[bid].x() +
                                                     d_unit_quaternion[bid].z() *d_unit_quaternion[bid].z() );
                        rot_matrix_acc[1][2] = 2 * (d_unit_quaternion[bid].y() * d_unit_quaternion[bid].z() - \
                                                d_unit_quaternion[bid].x() * d_unit_quaternion[bid].w());
                    }
                    else if( grp.get_local_id(0) == 2){
                        rot_matrix_acc[2][0] = 2 * (d_unit_quaternion[bid].x() * d_unit_quaternion[bid].z() - \
                                                d_unit_quaternion[bid].y() * d_unit_quaternion[bid].w());
                        rot_matrix_acc[2][1] = 2 * (d_unit_quaternion[bid].y() * d_unit_quaternion[bid].z() + \
                                                d_unit_quaternion[bid].x() * d_unit_quaternion[bid].w());
                        rot_matrix_acc[2][2] = 1 - 2 * ( d_unit_quaternion[bid].x() * d_unit_quaternion[bid].x() + \
                                                     d_unit_quaternion[bid].y() * d_unit_quaternion[bid].y() );
                    }
                    //Rotate the atoms
                    grp.barrier();
                    
                    int tidx = grp.get_local_id(0);
                    int index = tidx + bid*number_of_atoms;
                    int tmp = bid+1;
                    if(index < number_of_atoms*tmp && number_of_atoms*bid <= index ){
                        
                        res[index].atom_id = atoms[tidx].atom_id;

                        res[index].position.x() = atoms[tidx].position.x() * rot_matrix_acc[0][0] + \
                                                atoms[tidx].position.y() * rot_matrix_acc[0][1] + \
                                                atoms[tidx].position.z() * rot_matrix_acc[0][2] + passingPoint->x();

                        res[index].position.y() = atoms[tidx].position[0] * rot_matrix_acc[1][0] + \
                                                atoms[tidx].position[1] * rot_matrix_acc[1][1] + \
                                                atoms[tidx].position[2] * rot_matrix_acc[1][2] + passingPoint->y();

                        res[index].position.z() = atoms[tidx].position[0] * rot_matrix_acc[2][0] + \
                                                atoms[tidx].position[1] * rot_matrix_acc[2][1] + \
                                                atoms[tidx].position[2] * rot_matrix_acc[2][2] + passingPoint->z();
                    }
                    
                }

            });// end rotation parallel_for
        }); // end rotation submit
        
        try{
            q_gpu.wait_and_throw();
        }catch(cl::sycl::exception const& e){
            std::cout << "SYCL: synchronous exception occured, Rotation class " << __LINE__ << "\n"
            << e.what() << std::endl; 
        }
        
        //std::cout << "rotation line " << __LINE__ << std::endl;
        //q_gpu.memcpy(h_res, d_res, size_of_atoms * NUM_OF_BLOCKS*number_of_atoms);
        //q_gpu.wait();
    }// SYCL BUFFER Scope
    
    std::vector<std::vector<atom_st>> result_to_return;
    
    std::vector<atom_st> tmp;

    //std::cout << "rotation line " << __LINE__ << std::endl;
    //copy the results in order to free the memory and to pass the result to other functions for further usage
    for (int i = 0; i < NUM_OF_BLOCKS; i++){
        for (int c = atoms_st.size() * i; c < atoms_st.size() * (i + 1); c++){
            //tmp.push_back(h_res[c]);
            tmp.push_back(res[c]);
        }
        result_to_return.push_back(tmp);
        tmp.clear();
    }
    //std::cout << "rotation line " << __LINE__ << std::endl;
    
    
    //DEBUG PRINTING
    #ifndef NDEBUG
    for(int i = 0; i < NUM_OF_BLOCKS; i++){
        std::cout << "angle of rotation : " << i << std::endl;
        for (int c = 0; c < result_to_return[i].size(); c++){
            std::cout << result_to_return[i][c].atom_id << " ";
            std::cout << result_to_return[i][c].position.x() << " ";
            std::cout << result_to_return[i][c].position.y() << " ";
            std::cout << result_to_return[i][c].position.z() << std::endl;
        }
    }
    #endif

    //free memory allocated using sycl::malloc
    cl::sycl::free(passingPoint,q_gpu);
    //cl::sycl::free(d_unit_quaternion,q_gpu);
    cl::sycl::free(atoms,q_gpu);
    //cl::sycl::free(d_res,q_gpu);
    //cl::sycl::free(h_res,q_gpu);
    cl::sycl::free(res,q_gpu);
    
    return result_to_return;
}