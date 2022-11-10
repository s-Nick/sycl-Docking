
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/FileParsers.h>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/FileParseException.h>
#include <RDGeneral/BadFileException.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>
#include <chrono>

#include "math_constants.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include "helper.h"

#define NUM_OF_BLOCKS 360

using namespace RDKit;

/**
 * Struct used to keep track of the max result found.
 * It keeps track of the distance, the angle and the rotamer. From old version and for future expansions,
 * it keeps also track of the rotated positions of the first half of the molecule. 
 **/
struct max_value{
    double distance;
    int angle;
    Rotamer rt;
    //atom_st* rot_mol_fst_half;
    std::string mol_name;
};


/**
 * Compute the unit quaternion used in the computation of the rotation matrix.
 * Each thread compute one unit_quaternion.
 * 
 * @param res Array with the result.
 * @param quaternion Array containing the data of the vector, which the atoms must rotate around.
 **/
__global__ void compute_unit_quaternions(double4* res, double3 quaternion){

    int tid = threadIdx.x;// + blockIdx.x*gridDim.x;

    double x , y ,z;
    double angle;
    double sin_2 , cos_2;

    //compute the norm of the vector.
    double norm = norm3d(quaternion.x, quaternion.y,quaternion.z);
    if(tid < 360){
        x = quaternion.x/norm;
        y = quaternion.y/norm;
        z = quaternion.z/norm;
        angle = CUDART_PI/180 * tid;
        sin_2 = sin(angle/2);
        cos_2 = cos(angle/2);
        res[tid] = make_double4(x*sin_2, y*sin_2 , z*sin_2 , cos_2);//computed accordingly to quaternion explained in the report.
    }

}

void analyzeMolecule(max_value& max_dist, std::shared_ptr<RDKit::ROMol> mol){
    std::vector<Rotamer> rotamers;
    std::vector<atom_st> atoms;

    // Initialize the graph.
    Graph graph = Graph(mol->getNumAtoms());
    
    auto conf = mol->getConformer();
    
    std::cout << "number of bonds: " << mol->getNumBonds() << '\n';// mol2->getNumBonds() << '\n';

    if( !mol->getRingInfo()->isInitialized() ) {
        RDKit::MolOps::findSSSR( *mol );
    }

    //for( unsigned int i = 0; i < mol->getNumBonds() ; i++ ) {
    //    const RDKit::Bond *bond = mol->getBondWithIdx( i );
    //}

    // Get all the Bond in the mol and add the valid ones to the rotamers' vector.
    // Since the Bond in rings and the Double bond are not considerated useful for
    // the rotation, it discards them.
    auto conv_to_double3 = [](const RDGeom::Point3D& pos) {
        return make_double3(pos[0], pos[1], pos[2]);
    };

    for( unsigned int i = 0; i < mol->getNumBonds() ; i++ ) {
        const RDKit::Bond *bond = mol->getBondWithIdx( i );
        unsigned int startingAtom, endingAtom;
        startingAtom = bond->getBeginAtomIdx();
        endingAtom = bond->getEndAtomIdx();
        graph.addEdge(startingAtom,endingAtom);
        if( mol->getRingInfo()->numBondRings( bond->getIdx() )) {
            //continue;
            std::cout <<  "Bond " << bond->getIdx() << " is in a ring " << "stAtom: " << startingAtom << " endAtom: " << endingAtom << "\n";
        }
        else if(bond->getBondType() == RDKit::Bond::BondType::DOUBLE){
            //continue;
            std::cout <<  "Bond " << bond->getIdx() << " is a DOUBLE bond " << "stAtom: " << startingAtom << " endAtom: " << endingAtom << "\n";
        }
        else{
            unsigned int id = bond->getIdx();
            atom_st beginAtom{startingAtom, conv_to_double3(conf.getAtomPos(startingAtom))} ;
            atom_st endAtom{endingAtom, conv_to_double3(conf.getAtomPos(endingAtom))};
            rotamers.push_back(Rotamer(*bond,id, beginAtom, endAtom));
        }
    }


    // Add all the atoms to the atoms' vector
    for(auto atom : mol->atoms()){
        const uint id = atom->getIdx();
        atoms.push_back(atom_st{id,conv_to_double3(conf.getAtomPos(id))});
    }

    //Initialize the result storing structure.
    //max_value max_dist;
    //max_dist.distance = 0;


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
    for(auto rt : rotamers){

        bool analize;

        // Removing the analize edge/bond
        graph.removeEdge(rt.getBeginAtom().id, rt.getEndingAtom().id);

        // Compute the two halves of the splitted molecule.
        graph.DFSlinkedNode(rt.getBeginAtom().id, first_half);
        graph.DFSlinkedNode(rt.getEndingAtom().id, second_half);

        std::vector<atom_st> atoms_first_half;
        std::vector<atom_st> atoms_second_half;
        
        for(auto i: first_half)  atoms_first_half.push_back(atoms[i]);
        
        for(auto i : second_half) atoms_second_half.push_back(atoms[i]);
        /*
        max_value max_first_half;
        max_first_half.distance = 0;
        max_value max_second_half;
        max_second_half.distance = 0;
        */

        Rotation r;

        // If the bond split, create one half with only one atom. The bond is not a rotamer,
        // so I don't rotate around it and skip the computation.
        if(atoms_first_half.size() > 1 && second_half.size() > 1){
            
            analize = true;
            std::cout << "Checking rotamer: " << rt.getBond().getIdx() << " ";
            std::cout << "Starting Atom: " << rt.getBeginAtom().id << " Ending Atom: " << rt.getEndingAtom().id << " ";

            std::cout << "number of atom in first half: " << atoms_first_half.size() << "\n";

            std::vector<atom_st> distance_to_compute;
            double4* unit_quaternions;

            cudaMallocManaged(&unit_quaternions, 360*sizeof(double4));

            int deviceId;
            cudaGetDevice(&deviceId);

            cudaMemPrefetchAsync(unit_quaternions,360*sizeof(double4),deviceId);
            
            //Vector of the rotamer considered in the loop.
            double3 tmp_vector = rt.getVector();

            // The computatioin of the unit quaternion is done in parallel for all
            // the angle, launching the kernel with 360 threads, one for each angle.
            compute_unit_quaternions<<<1,360>>>(unit_quaternions,tmp_vector);

            cudaDeviceSynchronize();
            
            
            double max = 0;
            double* res = nullptr;
            
            for(int c = 0; c < 360; c += NUM_OF_BLOCKS ){
                
                std::vector<std::vector<atom_st>> rot_first_half;
                
                double3 tmp = rt.getBeginAtom().position;

                // Compute the rotation and storing the result
                rot_first_half = r.rotate_v5(c , atoms_first_half, tmp, unit_quaternions);

                // Add all the element of the vector of vectors in a single vector with all the atoms.
                // The atoms are in order of angle of rotation and every time is added the missing atoms
                // of the second half of the molecule, in order to compute the internal distance.
                for(int rotation = 0; rotation < NUM_OF_BLOCKS; ++rotation){
                    //cout << "main line " << __LINE__ << endl;
                    for(int i = 0; i < atoms_first_half.size(); i++){
                        distance_to_compute.push_back(rot_first_half[rotation][i]);
                    }
                    //cout << "main line " << __LINE__ << endl;
                    for(atom_st at : atoms_second_half){
                        distance_to_compute.push_back(at);
                    }
                }

                // Compute the internal distance, storing the result in res.
                res = distance_v3(distance_to_compute, atoms.size(), NUM_OF_BLOCKS);
                
                // Select the rotation that has the highest internal distance,
                // cycling through the results stored in res. 
                for(int i = 0; i < NUM_OF_BLOCKS; ++i) {
                    if(res[i] > max_first_half.distance) {
                        max_first_half.distance = res[i];
                        max_first_half.angle = c+i;
                        max_first_half.rt = rt;
                    }
                }
                distance_to_compute.clear();
                std::vector<atom_st>().swap(distance_to_compute);

                rot_first_half.clear();
                std::vector<std::vector<atom_st>>().swap(rot_first_half);

            }


            printf("Computed distance for the first part,\n");
            printf("the max distance compute is %lf with angle %d around rotamer: %d\n", \
                    max_first_half.distance, max_first_half.angle,max_first_half.rt.getBond().getIdx());
            
            cudaFree(unit_quaternions);
            cudaFree(res);
        }
        else{
            analize = false;
            printf("Checking rotamer %d ... ", rt.getBond().getIdx());
            printf("Too few atoms in the partition, rotamer not analized\n");
        }

        double total = max_first_half.distance + max_second_half.distance;

        if(total > max_dist.distance){
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
        graph.addEdge(rt.getBeginAtom().id,rt.getEndingAtom().id);
        if(analize)
            printf("For Rotamer %d, the max distance computed is: %lf,\n with a first angle: %d \n",\
                rt.getBond().getIdx(),total,max_first_half.angle);

    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

    std::cout << "duration time[ms]: " << duration.count() << "\n";


    //printf("For molecule named %s \n", mol->getProp<std::string>("_Name") );
    std::cout << "For molecule named " << mol->getProp<std::string>("_Name") << "\n";

    printf("The maximum distance computed is %lf\n", max_first_half.distance);
    
    printf("Computed with an angle of %d, around the rotamer %d\n",
        max_first_half.angle,max_first_half.rt.getBond().getIdx());
    
    return;
}

/**
 * Main function of the code. It parse the file and retrieve all the necessary data for the computation.
 * It takes as input the mol2 file that describe the molecule.
 **/
int main(int argc, char** argv){

    std::string mol_file = argv[1];
    char* mol_number_string = argv[2];
    //std::vector<Rotamer> rotamers;
    //std::vector<atom_st> atoms;
    //RWMol *m = Mol2FileToMol( mol_file );
    //std::shared_ptr<RDKit::ROMol>const  mol( RDKit::Mol2FileToMol( mol_file,true,false,CORINA,false ) );

    /**
     * The following initialization works with the aspirin's mol2 file provided by the Professor.
     * The declaration above works only with the file found online.
     */
    //std::shared_ptr<RDKit::ROMol>const  mol( RDKit::Mol2FileToMol( mol_file,false,true,CORINA,false ) );
    /**The next Line read the molecule removing the H atoms, it reduce the number of possible rotors
     *  for the aspirin and it seems to work, but idk with others molecules, so for now I keep
     * more rotores, but with the possible right solution.
     */
    //std::shared_ptr<RDKit::ROMol> mol( RDKit::Mol2FileToMol( mol_file,true,true,CORINA,false ) );

    
    std::ifstream molFileStream;
    molFileStream.open(mol_file, std::ios::in);

    std::vector<std::shared_ptr<RDKit::ROMol>> molecules;
    //readMoleculesStream(molFileStream, molecules);
    int mol_number = atoi(mol_number_string);
    if(mol_number == 1){
        singleMoleculeRead(molFileStream, molecules);
    }
    else{
        multipleMoleculeRead(molFileStream,molecules);
    }
    //auto tmp = molecules[3]->getProp<std::string>("_Name");
    
    //Initialize the result storing structure.
    max_value max_dist;
    max_dist.distance = 0;


    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << molecules.size() << "\n";
    
    
    for(auto mol : molecules){
        analyzeMolecule(max_dist, mol);
    }
    
    
    auto final_stop = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(final_stop-total_start);

    std::cout << "total duration of the computation[ms]: " << total_duration.count() << "\n"; 

    printf("The overall maximum distance computed is %lf ", max_dist.distance);
    std::cout  << " obtained from molecule " <<  max_dist.mol_name << "\n"; 
        
    printf("Computed with an angle of %d, around the rotamer %d\n",max_dist.angle,max_dist.rt.getBond().getIdx());
    
    return 0;
}