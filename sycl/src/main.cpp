
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/FileParsers.h>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/FileParseException.h>
#include <RDGeneral/BadFileException.h>

#include<CL/sycl.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>
#include <chrono>

#include"helper.h"



#define NUM_OF_BLOCKS 360

using namespace std;

/**
 * Struct used to keep track of the max result found.
 * It keeps track of the distance, the angle and the rotamer. From old version and for future expansions,
 * it keeps also track of the rotated positions of the first half of the molecule. 
 **/
struct max_value
{
    double distance;
    int angle;
    Rotamer rt;
    atom_st *rot_mol_fst_half;
};


class computeQuaternions;
//SYCL function to compute unit_quaternions on gpu and return them.
void computeUnitQuaternions(cl::sycl::double4* results, cl::sycl::double3  rt_vector){

    {
        cl::sycl::device gpu = cl::sycl::gpu_selector{}.select_device();
        cl::sycl::queue q(gpu);

        cl::sycl::double4 *gpu_result = 
            cl::sycl::malloc_shared<cl::sycl::double4>(360*sizeof(cl::sycl::double4),q);

        //Check if allocate enough memory
        //cl::sycl::double3 gpu_rt_vector;
        cl::sycl::double3* gpu_rt_vector =
            cl::sycl::malloc_shared<cl::sycl::double3>(sizeof(cl::sycl::double3),q);
        
        //Initialize gpu_rt_vector copying the data from cpu memory
        gpu_rt_vector->x() = rt_vector.x();
        gpu_rt_vector->y() = rt_vector.y();
        gpu_rt_vector->z() = rt_vector.z();
        
        /*
        for(int i = 0; i < rt_vector.get_size(); i++){
            gpu_rt_vector[i] = rt_vector[i];

        }
        */
        

        //q.submit([=] (cl::sycl::handler & cgh){
            //cl::sycl::stream out(2048, 256, cgh);
            q.parallel_for<class computeQuaternions>(360, [=](cl::sycl::id<1> tid){
                //cl::sycl::double3 norm;
                double norm;
                double x, y, z;
                double angle;
                double sin_2, cos_2;
                
                //TEMPORARY SOLUTION TO FIX MEMORY
                //cl::sycl::double3 gpu_rt_vector = gpu_rt_vector_shared[0];
                
                double x_tmp_2, y_tmp_2, z_tmp_2;
                norm = cl::sycl::length(cl::sycl::double3(gpu_rt_vector->x(),
                                                    gpu_rt_vector->y(),gpu_rt_vector->z()));



                if(tid < 360){
                    x = gpu_rt_vector->x()/norm;
                    y = gpu_rt_vector->y()/norm;
                    z = gpu_rt_vector->z()/norm;
                    angle = M_PI/180 * tid;
                    sin_2 = cl::sycl::sin(angle/2);
                    cos_2 = cl::sycl::cos(angle/2);
                    gpu_result[tid] = cl::sycl::double4{x*sin_2 , y*sin_2, z*sin_2, cos_2};
                }
            }).wait();
        //}).wait();

        for(int i = 0; i < 360; i++){
            results[i] = gpu_result[i];
        }

        cl::sycl::free(gpu_rt_vector,q);
        cl::sycl::free(gpu_result,q);
    }
    return;

}


/**
 * Main function of the code. It parse the file and retrieve all the necessary data for the computation.
 * It takes as input the mol2 file that describe the molecule.
 **/
int main(int argc, char **argv)
{

    std::string mol_file = argv[1];

    std::vector<Rotamer> rotamers;
    std::vector<atom_st> atoms;
    //RWMol *m = Mol2FileToMol( mol_file );
    //std::shared_ptr<RDKit::ROMol>const  mol( RDKit::Mol2FileToMol( mol_file,true,false,CORINA,false ) );

    /**
     * The following initialization works with the aspirin's mol2 file provided by the Professor.
     * The declaration above works only with the file found online.
     */
    std::shared_ptr<RDKit::ROMol> const mol(RDKit::Mol2FileToMol(mol_file, false, true, RDKit::CORINA, false));
    /**The next Line read the molecule removing the H atoms, it reduce the number of possible rotors
     *  for the aspirin and it seems to work, but idk with others molecules, so for now I keep
     * more rotores, but with the possible right solution.
     */
    //std::shared_ptr<RDKit::ROMol> mol( RDKit::Mol2FileToMol( mol_file,true,true,CORINA,false ) );

    // Initialize the graph.
    Graph graph = Graph(mol->getNumAtoms());

    auto conf = mol->getConformer();

    std::cout << "number of bonds: " << mol->getNumBonds() << '\n'; // mol2->getNumBonds() << '\n';

    if (!mol->getRingInfo()->isInitialized())
    {
        RDKit::MolOps::findSSSR(*mol);
    }

    //for( unsigned int i = 0; i < mol->getNumBonds() ; i++ ) {
    //    const RDKit::Bond *bond = mol->getBondWithIdx( i );
    //}

    // Get all the Bond in the mol and add the valid ones to the rotamers' vector.
    // Since the Bond in rings and the Double bond are not considerated useful for
    // the rotation, it discards them.
    for (unsigned int i = 0; i < mol->getNumBonds(); i++)
    {
        const RDKit::Bond *bond = mol->getBondWithIdx(i);
        unsigned int startingAtom, endingAtom;
        startingAtom = bond->getBeginAtomIdx();
        endingAtom = bond->getEndAtomIdx();
        graph.addEdge(startingAtom, endingAtom);
        if (mol->getRingInfo()->numBondRings(bond->getIdx()))
        {
            //continue;
            std::cout << "Bond " << bond->getIdx() << " is in a ring "
                      << "stAtom: " << startingAtom << " endAtom: " << endingAtom << endl;
        }
        else if (bond->getBondType() == RDKit::Bond::BondType::DOUBLE)
        {
            //continue;
            std::cout << "Bond " << bond->getIdx() << " is a DOUBLE bond "
                      << "stAtom: " << startingAtom << " endAtom: " << endingAtom << endl;
        }
        else
        {
            unsigned int id = bond->getIdx();
            atom_st beginAtom;
            atom_st endAtom;
            beginAtom.atom_id = startingAtom;
            endAtom.atom_id = endingAtom;
            auto tmp_pos = conf.getAtomPos(beginAtom.atom_id);
            beginAtom.position = cl::sycl::double3{tmp_pos[0], tmp_pos[1], tmp_pos[2]};
            tmp_pos = conf.getAtomPos(endAtom.atom_id);
            endAtom.position = cl::sycl::double3{tmp_pos[0], tmp_pos[1], tmp_pos[2]};
            Rotamer rt = Rotamer(*bond, id, beginAtom, endAtom);
            rotamers.push_back(rt);
        }
    }

    // Add all the atoms to the atoms' vector
    for (auto atom : mol->atoms())
    {
        uint id = atom->getIdx();
        auto pos_tmp = conf.getAtomPos(id);
        cl::sycl::double3 pos = cl::sycl::double3{pos_tmp[0], pos_tmp[1], pos_tmp[2]};
        atom_st at;
        at.atom_id = id;
        at.position = pos;
        atoms.push_back(at);
    }

    //Initialize the result storing structure.
    max_value max_dist;
    max_dist.distance = 0;

    vector<unsigned int> first_half;
    vector<unsigned int> second_half;
    //Rotamer rt = rotamers[0];
    //vector<Rotamer> tmp_rotamers ={rotamers[0], rotamers[1]};
    auto start = std::chrono::high_resolution_clock::now();
    // Cycle through all the available rotamers
    for (auto rt : rotamers){

        bool analize;

        // Removing the analize edge/bond
        graph.removeEdge(rt.getBeginAtom().atom_id, rt.getEndingAtom().atom_id);

        // Compute the two halves of the splitted molecule.
        graph.DFSlinkedNode(rt.getBeginAtom().atom_id, first_half);
        graph.DFSlinkedNode(rt.getEndingAtom().atom_id, second_half);

        vector<atom_st> atoms_first_half;
        vector<atom_st> atoms_second_half;

        for (auto i : first_half)
            atoms_first_half.push_back(atoms[i]);

        for (auto i : second_half)
            atoms_second_half.push_back(atoms[i]);

        max_value max_first_half;
        max_first_half.distance = 0;
        max_value max_second_half;
        max_second_half.distance = 0;

        Rotation r;

        // If the bond split, create one half with only one atom. The bond is not a rotamer,
        // so I don't rotate around it and skip the computation.
        if (atoms_first_half.size() > 1 && second_half.size() > 1){

            analize = true;
            std::cout << "Checking rotamer: " << rt.getBond().getIdx() << " ";
            std::cout << "Starting Atom: " << rt.getBeginAtom().atom_id << " Ending Atom: " << rt.getEndingAtom().atom_id << " ";

            std::cout << "number of atom in first half: " << atoms_first_half.size() << endl;

            vector<atom_st> distance_to_compute;
            
            cl::sycl::double3 tmp_vector = rt.getVector();
            
            cl::sycl::double4* unit_quaternions = (cl::sycl::double4*)malloc(360*sizeof(cl::sycl::double4));

            computeUnitQuaternions(unit_quaternions,tmp_vector);
            /*
            //DEBUG PRINTING
            for(int i = 0; i < NUM_OF_BLOCKS; i++){
                std::cout << " unit quaternion of rotation : " << i << std::endl;
                for (int c = 0; c < 4; c++){
                    std::cout << unit_quaternions[i][c] << " ";
                }
                std::cout << std::endl;
            }
            */
            //cout << "UNIT QUATERNION TEST: " << unit_quaternions[0][0] << " " <<__LINE__ << endl;

            double max = 0;
            double *res;

            for (int c = 0; c < 360; c += NUM_OF_BLOCKS){

                vector<vector<atom_st>> rot_first_half;

                cl::sycl::double3 tmp = rt.getBeginAtom().position;

                // Compute the rotation and storing the result
                rot_first_half = r.rotate(c, atoms_first_half, tmp, unit_quaternions);
                
                
                // Add all the element of the vector of vectors in a single vector with all the atoms.
                // The atoms are in order of angle of rotation and every time is added the missing atoms
                // of the second half of the molecule, in order to compute the internal distance.
                
                for (int rotation = 0; rotation < NUM_OF_BLOCKS; rotation++){
                    //cout << "main line " << __LINE__ << endl;
                    for (int i = 0; i < atoms_first_half.size(); i++){
                        distance_to_compute.push_back(rot_first_half[rotation][i]);
                    }
                    //cout << "main line " << __LINE__ << endl;
                    for (atom_st at : atoms_second_half){
                        distance_to_compute.push_back(at);
                    }
                }
                
                // Compute the internal distance, storing the result in res.
                res = distance(distance_to_compute, atoms.size(), NUM_OF_BLOCKS);

                

                for (int i = 0; i < NUM_OF_BLOCKS; i++){
                    if (res[i] > max_first_half.distance){
                        max_first_half.distance = res[i];
                        max_first_half.angle = c + i;
                        max_first_half.rt = rt;
                    }
                }
                
                distance_to_compute.clear();
                
                //rot_first_half.clear();
                
                

            }
            printf("Computed distance for the first part,\n");
            printf("the max distance compute is %lf with angle %d around rotamer: %d\n",
                   max_first_half.distance, max_first_half.angle, max_first_half.rt.getBond().getIdx());

            std::free(unit_quaternions);
            
            /*
            {//SYCL buffer scope for unit_quaternions

                cl::sycl::buffer<cl::sycl::double4, 1> buf_unit_quaternion(unit_quaternions,cl::sycl::range<1>(1));

            
            }// end scope unit_quaternion buffer
            */
        }
        else{
            analize = false;
            printf("Checking rotamer %d ... ", rt.getBond().getIdx());
            printf("Too few atoms in the partition, rotamer not analized\n");
        }
        
        double total = max_first_half.distance + max_second_half.distance;

        if (total > max_dist.distance){
            max_dist.distance = total;
            max_dist.rt = max_first_half.rt;
            max_dist.angle = max_first_half.angle;
        }
        
        first_half.clear();
        second_half.clear();
        atoms_first_half.clear();
        atoms_second_half.clear();

        // Adding again the edge corresponding to the bond, before computing another bond/rotamer.
        graph.addEdge(rt.getBeginAtom().atom_id, rt.getEndingAtom().atom_id);
        
        if (analize)
            printf("For Rotamer %d, the max distance computed is: %lf,\n with a first angle: %d \n",
                   rt.getBond().getIdx(), total, max_first_half.angle);
        
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "duration time[ms]: " << duration.count() << endl;

    printf("The maximum distance computed is %lf\n", max_dist.distance);

    printf("Computed with an angle of %d, around the rotamer %d\n", max_dist.angle, max_dist.rt.getBond().getIdx());

    return 0;
}