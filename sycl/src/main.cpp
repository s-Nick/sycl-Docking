
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

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

#define NUM_OF_BLOCKS 360

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
    //atom_st *rot_mol_fst_half;
    std::string mol_name;
};


class computeQuaternions;
//SYCL function to compute unit_quaternions on gpu and return them.
void computeUnitQuaternions(cl::sycl::double4* gpu_result, cl::sycl::double3  rt_vector, cl::sycl::queue& q_gpu){

    {
        cl::sycl::double3* gpu_rt_vector =
            cl::sycl::malloc_shared<cl::sycl::double3>(sizeof(cl::sycl::double3),q_gpu);
        
        //Initialize gpu_rt_vector copying the data from cpu memory
        gpu_rt_vector->x() = rt_vector.x();
        gpu_rt_vector->y() = rt_vector.y();
        gpu_rt_vector->z() = rt_vector.z();
        
        
        q_gpu.parallel_for<class computeQuaternions>(360, [=](cl::sycl::id<1> tid){

            double x, y, z;
            double angle;
            double sin_2, cos_2;
            
            double norm = cl::sycl::length(cl::sycl::double3(gpu_rt_vector->x(),
                                                gpu_rt_vector->y(),gpu_rt_vector->z()));

            if(tid < 360){
                x = gpu_rt_vector->x()/norm;
                y = gpu_rt_vector->y()/norm;
                z = gpu_rt_vector->z()/norm;
                angle = M_PI/180 * tid.get(0);
                sin_2 = cl::sycl::sin(angle/2);
                cos_2 = cl::sycl::cos(angle/2);
                gpu_result[tid] = cl::sycl::double4{x*sin_2 , y*sin_2, z*sin_2, cos_2};
            }
        }).wait();
        
        cl::sycl::free(gpu_rt_vector,q_gpu);
    }
    return;

}


/**
 * Main function of the code. It parse the file and retrieve all the necessary data for the computation.
 * It takes as input the mol2 file that describe the molecule.
 **/
int main(int argc, char **argv)
{

    std::string mol_file;
    bool multple_mol = false;

    po::options_description d;

    d.add_options()
    ("mol2-file,f", po::value(&mol_file)->required(), "The file containing the molecules")
    ("multpile molecules in files,n", po::value(&multple_mol)->required(), "if the molecule in the file is a single one o multiple")
    ;

    po::variables_map vm;
    po::options_description app_description("Executable options");
    app_description.add(d);
    po::store(po::command_line_parser(argc, argv).options(app_description).run() ,vm);
    po::notify(vm);
    // */
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

    std::vector<Rotamer> rotamers;
    std::vector<atom_st> atoms;

    /**
     * The following initialization works with the aspirin's mol2 file provided by the Professor.
     * The declaration above works only with the file found online.
     */
    //std::shared_ptr<RDKit::ROMol> const mol(RDKit::Mol2FileToMol(mol_file, false, true, RDKit::CORINA, false));
    /**The next Line read the molecule removing the H atoms, it reduce the number of possible rotors
     *  for the aspirin and it seems to work, but idk with others molecules, so for now I keep
     * more rotores, but with the possible right solution.
     */
    //std::shared_ptr<RDKit::ROMol> mol( RDKit::Mol2FileToMol( mol_file,true,true,CORINA,false ) );

    std::ifstream molFileStream;
    molFileStream.open(mol_file, std::ios::in);

    std::vector<std::shared_ptr<RDKit::ROMol>> molecules;
    //readMoleculesStream(molFileStream, molecules);
    // int mol_number = atoi(mol_number_string);
    if(!multple_mol){
        singleMoleculeRead(molFileStream, molecules);
    }
    else{
        multipleMoleculeRead(molFileStream,molecules);
    }
    std::cout << __LINE__ << std::endl;
    //Initialize the result storing structure.
    max_value max_dist;
    max_dist.distance = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    //Create the queue for the device in order to have the same context
    cl::sycl::device gpu = cl::sycl::gpu_selector{}.select_device();
    cl::sycl::queue q_gpu(gpu,exception_handler, cl::sycl::property::queue::in_order());

    for(auto mol : molecules){
        // Initialize the graph.
        Graph graph = Graph(mol->getNumAtoms());

        auto conf = mol->getConformer();

        std::cout << "number of bonds: " << mol->getNumBonds() << '\n'; // mol2->getNumBonds() << '\n';

        if (!mol->getRingInfo()->isInitialized())
        {
            RDKit::MolOps::findSSSR(*mol);
        }

        // Get all the Bond in the mol and add the valid ones to the rotamers' vector.
        // Since the Bond in rings and the Double bond are not considerated useful for
        // the rotation, it discards them.
        auto conv_to_double3 = [](const RDGeom::Point3D& pos) {
            return cl::sycl::double3(pos[0], pos[1], pos[2]);
        };

        for (unsigned int i = 0; i < mol->getNumBonds(); i++)
        {
            const RDKit::Bond *bond = mol->getBondWithIdx(i);
            unsigned int startingAtom, endingAtom;
            startingAtom = bond->getBeginAtomIdx();
            endingAtom = bond->getEndAtomIdx();
            graph.addEdge(startingAtom, endingAtom);
            if (mol->getRingInfo()->numBondRings(bond->getIdx()))
            {
                #ifndef NDEBUG
                std::cout << "Bond " << bond->getIdx() << " is in a ring "
                        << "stAtom: " << startingAtom << " endAtom: " << endingAtom << '\n';
                #endif
                continue;
            }
            else if (bond->getBondType() == RDKit::Bond::BondType::DOUBLE)
            {
                #ifndef NDEBUG
                std::cout << "Bond " << bond->getIdx() << " is a DOUBLE bond "
                        << "stAtom: " << startingAtom << " endAtom: " << endingAtom << '\n';
                #endif
                continue;
            }
            else
            {
                unsigned int id{bond->getIdx()};
                atom_st beginAtom{startingAtom, conv_to_double3(conf.getAtomPos(startingAtom))};
                atom_st endAtom{endingAtom, conv_to_double3(conf.getAtomPos(endingAtom))};
                rotamers.push_back(Rotamer(*bond, id, beginAtom, endAtom));
            }
        }

        // Add all the atoms to the atoms' vector
        for (auto atom : mol->atoms())
        {
            const uint id = atom->getIdx();
            atoms.push_back(atom_st{ id, conv_to_double3(conf.getAtomPos(id)) });
        }

        max_value max_first_half;
        max_first_half.distance = 0;
        max_value max_second_half;
        max_second_half.distance = 0;
            

        std::vector<unsigned int> first_half;
        std::vector<unsigned int> second_half;
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

            std::vector<atom_st> atoms_first_half;
            std::vector<atom_st> atoms_second_half;

            for (auto i : first_half)
                atoms_first_half.push_back(atoms[i]);

            for (auto i : second_half)
                atoms_second_half.push_back(atoms[i]);

            /*
            max_value max_first_half;
            max_first_half.distance = 0;
            max_value max_second_half;
            max_second_half.distance = 0;
            */
            Rotation r;
            // If the bond split, create one half with only one atom. The bond is not a rotamer,
            // so I don't rotate around it and skip the computation.
            if (atoms_first_half.size() > 1 && second_half.size() > 1){

                analize = true;
                #ifndef NDEBUG
                std::cout << "Checking rotamer: " << rt.getBond().getIdx() << " ";
                std::cout << "Starting Atom: " << rt.getBeginAtom().atom_id << " Ending Atom: " << rt.getEndingAtom().atom_id << " ";
                std::cout << "number of atom in first half: " << atoms_first_half.size() << '\n';
                #endif

                std::vector<atom_st> distance_to_compute;
                
                cl::sycl::double3 tmp_vector = rt.getVector();
                
                //cl::sycl::double4* unit_quaternions = (cl::sycl::double4*)malloc(360*sizeof(cl::sycl::double4));

                cl::sycl::double4* unit_quaternions = cl::sycl::malloc_shared<cl::sycl::double4>(sizeof(cl::sycl::double4)*NUM_OF_BLOCKS, q_gpu);

                computeUnitQuaternions(unit_quaternions, tmp_vector, q_gpu);
                // q_gpu.wait();
                //cout << "UNIT QUATERNION TEST: " << unit_quaternions[0][0] << " " <<__LINE__ << endl;
                
                double max = 0;
                double *res = nullptr;

                for (int c = 0; c < 360; c += NUM_OF_BLOCKS){

                    std::vector<std::vector<atom_st>> rot_first_half;

                    cl::sycl::double3 tmp = rt.getBeginAtom().position;

                    // Compute the rotation and storing the result
                    rot_first_half = r.rotate(c, atoms_first_half, tmp, unit_quaternions, q_gpu);
                    
                    // Add all the element of the vector of vectors in a single vector with all the atoms.
                    // The atoms are in order of angle of rotation and every time is added the missing atoms
                    // of the second half of the molecule, in order to compute the internal distance.
                    
                    for (int rotation = 0; rotation < NUM_OF_BLOCKS; ++rotation){
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
                    res = distance(distance_to_compute, atoms.size(), NUM_OF_BLOCKS, q_gpu);

                    // q_gpu.wait();

                    for (int i = 0; i < NUM_OF_BLOCKS; i++){
                        if (res[i] > max_first_half.distance){
                            max_first_half.distance = res[i];
                            max_first_half.angle = c + i;
                            max_first_half.rt = rt;
                        }
                    }
                    
                    distance_to_compute.clear();
                    std::vector<atom_st>().swap(distance_to_compute);

                    rot_first_half.clear();
                    std::vector<std::vector<atom_st>>().swap(rot_first_half);

                    

                }
                #ifndef NDEBUG
                printf("Computed distance for the first part,\n");
                printf("the max distance compute is %lf with angle %d around rotamer: %d\n",
                    max_first_half.distance, max_first_half.angle, max_first_half.rt.getBond().getIdx());
                #endif
                cl::sycl::free(unit_quaternions,q_gpu);
                //std::free(unit_quaternions);
                std::free(res);
            }
            else{
                analize = false;
                #ifndef NDEBUG
                printf("Checking rotamer %d ... ", rt.getBond().getIdx());
                printf("Too few atoms in the partition, rotamer not analized\n");
                #endif
            }
            
            double total = max_first_half.distance; //+ max_second_half.distance;

            if (total > max_dist.distance){
                max_dist.distance = total;
                max_dist.rt = max_first_half.rt;
                max_dist.angle = max_first_half.angle;
                max_dist.mol_name = mol->getProp<std::string>("_Name");
            }
            
            first_half.clear();
            second_half.clear();
            atoms_first_half.clear();
            atoms_second_half.clear();

            // Adding again the edge corresponding to the bond, before computing another bond/rotamer.
            graph.addEdge(rt.getBeginAtom().atom_id, rt.getEndingAtom().atom_id);

            #ifndef NDEBUG 
            if (analize)
                printf("For Rotamer %d, the max distance computed is: %lf,\n with a first angle: %d \n",
                    rt.getBond().getIdx(), total, max_first_half.angle);
            #endif 
        }
    
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        #ifndef NDEBUG
        std::cout << "duration time[ms]: " << duration.count() << '\n';
        std::cout << "For molecule named " << mol->getProp<std::string>("_Name") << '\n';
        #endif
        printf("The maximum distance computed is %lf\n", max_first_half.distance);

        printf("Computed with an angle of %d, around the rotamer %d\n", 
            max_first_half.angle, max_first_half.rt.getBond().getIdx());

        rotamers.clear();
        atoms.clear();

    }

    auto final_stop = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(final_stop-total_start);

    std::cout << "total duration of the computation[ms]: " << total_duration.count() << std::endl; 

    printf("The overall maximum distance computed is %lf ", max_dist.distance);
    std::cout  << " obtained from molecule " <<  max_dist.mol_name << std::endl;
        
    printf("Computed with an angle of %d, around the rotamer %d\n",max_dist.angle,max_dist.rt.getBond().getIdx());
    

    return 0;
}