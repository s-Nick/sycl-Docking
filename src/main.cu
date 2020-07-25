/**
* 25/06 done some experiment on extracting information from the molecule. Get
all the position onf atoms and which bonds are in a ring and which bonds are double.
need to do all the parallel computation and rotation.
***/
#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/FileParsers.h>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/FileParseException.h>
#include <RDGeneral/BadFileException.h>

#include <algorithm>
#include <string>
#include "math_constants.h"
#include <stdio.h>
#include <chrono>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include <vector>

#include "atom.cuh"
#include "rotamer.cuh"
#include "graph.h"
#include "distance.h"

#define NUM_OF_STREAMS 360

using namespace RDKit;
using namespace std;

struct max_value{
    double distance;
    int angle;
    Rotamer rt;
    atom_st* rot_mol_fst_half;
};

__global__ void compute_unit_quaternions(double4* res, double3 quaternion){

    int tid = threadIdx.x;// + blockIdx.x*gridDim.x;

    double norm;
    double x , y ,z;
    double angle;
    double sin_2 , cos_2;


    norm = norm3d(quaternion.x, quaternion.y,quaternion.z);
    if(tid < 360){
        x = quaternion.x/norm;
        y = quaternion.y/norm;
        z = quaternion.z/norm;
        angle = CUDART_PI/180 * tid;
        sin_2 = sin(angle/2);
        cos_2 = cos(angle/2);
        res[tid] = make_double4(x*sin_2, y*sin_2 , z*sin_2 , cos_2);
    }

}

int main(int argc, char** argv){

    std::string mol_file = argv[1];
    //RWMol *m = Mol2FileToMol( mol_file );
    //std::shared_ptr<RDKit::ROMol>const  mol( RDKit::Mol2FileToMol( mol_file,true,false,CORINA,false ) );

    /**
     * The following initialization works with the aspirin's mol2 file provided by the Professor.
     * The declaration above works only with the file found online.
     */
    std::shared_ptr<RDKit::ROMol>const  mol( RDKit::Mol2FileToMol( mol_file,false,true,CORINA,false ) );
    /**The next Line read the molecule removing the H atoms, it reduce the number of possible rotors
     *  for the aspirin and it seems to work, but idk with others molecules, so for now I keep
     * more rotores, but with the possible right solution.
     */
    //std::shared_ptr<RDKit::ROMol> mol( RDKit::Mol2FileToMol( mol_file,true,true,CORINA,false ) );

    //std::cout << *mol2 << std::endl;

    std::shared_ptr<RDKit::ROMol> mol2 = mol;


    Graph graph = Graph(mol->getNumAtoms());
    //std::shared_ptr<RDKit::ROMol> mol2(RDKit::MolOps::addHs(*mol,true,true,NULL,true ));

    auto conf = mol2->getConformer();
/*
    for(auto atom: mol2->atoms()) {
        std::cout << atom->getAtomicNum() << " ";
    }
        std::cout << std::endl;

    for(auto bond: mol2->bonds()) {
        std::cout << bond->getBondType() << " ";
        }
        std::cout << std::endl;
    for( unsigned int i = 0 , is = mol2->getNumBonds() ; i < is ; ++i ) {
        const RDKit::Bond *bond = mol2->getBondWithIdx( i );
        std::cout << bond->getIsAromatic() << " ";
    }
        std::cout << std::endl;
*/
    std::cout << "number of bonds: " << mol2->getNumBonds() << '\n';
    //std::cout << "number of atoms: " << mol->atoms() << '\n';

    if( !mol2->getRingInfo()->isInitialized() ) {
        RDKit::MolOps::findSSSR( *mol2 );
    }

    for( unsigned int i = 0; i < mol->getNumBonds() ; i++ ) {
        const RDKit::Bond *bond = mol->getBondWithIdx( i );
    }

    std::vector<Rotamer> rotamers;

    for( unsigned int i = 0; i < mol->getNumBonds() ; i++ ) {
        const RDKit::Bond *bond = mol->getBondWithIdx( i );
        unsigned int startingAtom, endingAtom;
        startingAtom = bond->getBeginAtomIdx();
        endingAtom = bond->getEndAtomIdx();
        graph.addEdge(startingAtom,endingAtom);
        //cout << "stAtom: " << startingAtom << " endAtom: " << endingAtom << endl;
        //RDKit::Bond *bond = mol->getBondWithIdx( i );
        if( mol->getRingInfo()->numBondRings( bond->getIdx() )) {
            //continue;
            std::cout <<  "Bond " << bond->getIdx() << " is in a ring " << "stAtom: " << startingAtom << " endAtom: " << endingAtom << endl;
        }
        else if(bond->getBondType() == RDKit::Bond::BondType::DOUBLE){
            //continue;
            std::cout <<  "Bond " << bond->getIdx() << " is a DOUBLE bond " << "stAtom: " << startingAtom << " endAtom: " << endingAtom << endl;
        }
        else{
            unsigned int id = bond->getIdx();
            atom_st beginAtom;
            atom_st endAtom;
            beginAtom.id = startingAtom;
            endAtom.id = endingAtom;
            auto tmp_pos = conf.getAtomPos(beginAtom.id);
            beginAtom.position = make_double3(tmp_pos[0],tmp_pos[1],tmp_pos[2]);
            tmp_pos = conf.getAtomPos(endAtom.id);
            endAtom.position = make_double3(tmp_pos[0],tmp_pos[1],tmp_pos[2]);
            Rotamer rt = Rotamer(*bond,id, beginAtom, endAtom);
            rotamers.push_back(rt);
        }
    }

    /*
    for(Rotamer &rt : rotamers){
        RDKit::Bond bond = rt.getBond();
        std::cout << bond.getBeginAtomIdx() << ' ' << bond.getEndAtomIdx() << std::endl;
        float3 bPos = rt.getBeginAtom().position;
        std::cout << "Prova " << rt.getBeginAtom().id << ' ' << bPos.x << ' ' << bPos.y << ' ' << bPos.z << endl;
    }

    for(RDKit::Bond &bond : rotamers){
        std::cout << bond.getBeginAtomIdx() << ' ' << bond.getEndAtomIdx() << std::endl;
    }*/


    //std::vector<RDGeom::Point3D> positions;
    //vector<float3> positions;
    // if the Point3D create problem with the gpu use thi method
    //vector<vector<double>> positions;

    //auto pos = conf.getAtomPos(0);
    //std::cout << typeid(pos).name() << std::endl;
    //std::cout << pos[0] << ' ' << pos[1] << ' ' << pos[2] << std::endl;

    std::vector<atom_st> atoms;

    for(auto atom : mol2->atoms()){
        uint id = atom->getIdx();
        auto pos_tmp = conf.getAtomPos(id);
        //vector<double> tmp{pos[0],pos[1],pos[2]};
        double3 pos = make_double3(pos_tmp[0],pos_tmp[1],pos_tmp[2]);
        atom_st at;
        at.id = id;
        at.position = pos;
        atoms.push_back(at);
        //positions.push_back(pos);
        //cout << "atom id: " << id << " position: " << pos.x << ' ' << pos.y << ' ' << pos.z << std::endl;
    }


    max_value max_dist;
    max_dist.distance = 0;


    vector<unsigned int> first_half;
    vector<unsigned int> second_half;
    //Rotamer rt = rotamers[0];
    //vector<Rotamer> tmp_rotamers ={rotamers[0], rotamers[1]};
    auto start = std::chrono::high_resolution_clock::now();
    for(auto rt : rotamers){
        graph.removeEdge(rt.getBeginAtom().id, rt.getEndingAtom().id);

        graph.DFSlinkedNode(rt.getBeginAtom().id, first_half);
        graph.DFSlinkedNode(rt.getEndingAtom().id, second_half);

        vector<atom_st> atoms_first_half;
        vector<atom_st> atoms_second_half;
        //atom_st* atoms_second_half[second_half.size()];
        for(auto i: first_half)  atoms_first_half.push_back(atoms[i]);
        
        //for(int i = 0; i < second_half.size(); i++){
          //  atoms_second_half[i] = &atoms[second_half[i]];
        //}
        for(auto i : second_half) atoms_second_half.push_back(atoms[i]);


        max_value max_first_half;
        max_first_half.distance = 0;
        max_value max_second_half;
        max_second_half.distance = 0;

        Rotation r = Rotation(rt.getVector());
        if(atoms_first_half.size() > 1 && second_half.size() > 1){

            cout << "Checking rotamer: " << rt.getBond().getIdx() << " ";
            cout << "Starting Atom: " << rt.getBeginAtom().id << " Ending Atom: " << rt.getEndingAtom().id << " ";

            cout << "number of atom in first half: " << atoms_first_half.size() << endl;

            //vector<atom_st> distance_to_compute;
            vector<atom_st> distance_to_compute;
            double4* unit_quaternions;

            cudaMallocManaged(&unit_quaternions, 2*360*sizeof(double4));

            int deviceId;
            cudaGetDevice(&deviceId);

            //cudaMemPrefetchAsync(unit_quaternions,2*360*sizeof(double4),deviceId);

            double3 tmp_vector = rt.getVector();

            compute_unit_quaternions<<<1,360>>>(unit_quaternions,tmp_vector);

            cudaDeviceSynchronize();
            /*
            for(int i = 0; i < 360; i++){
                printf("uq %d pos: %lf x %lf y %lf z %lf w\n", i, unit_quaternions[i].x, unit_quaternions[i].y,unit_quaternions[i].z,unit_quaternions[i].w);
            }*/
            
            double max = 0;
            double* res;
            //cout << "main line " << __LINE__ << endl;
            for(int c = 0; c < 360; c += NUM_OF_STREAMS ){
                
                //vector<atom_st*> rot_first_half;
                vector<vector<atom_st>> rot_first_half;
                //atom_st* rot_first_half;
                //double angle = (CUDART_PI/180) * (c);
                double3 tmp = rt.getBeginAtom().position;

                //printf("angle %lf\n",angle);
                //rot_first_half = r.rotate(angle , atoms_first_half, tmp);
                //rot_first_half = r.rotate_v2(c , atoms_first_half, tmp, unit_quaternions[c]);
                //cout << "main line " << __LINE__ << endl;
                rot_first_half = r.rotate_v5(c , atoms_first_half, tmp, unit_quaternions);

                /*
                for(atom_st at : atoms_first_half){
                    cout << at.position.x << " " << at.position.y << " "<< at.position.z << endl;
                }*/

                //cout << "main line " << __LINE__ << endl;
                //atom_st distance_to_compute[atoms.size()];
                //cout << "main line " << __LINE__ << endl;
                for(int rotation = 0; rotation < NUM_OF_STREAMS; rotation++){
                    //cout << "main line " << __LINE__ << endl;
                    for(int i = 0; i < atoms_first_half.size(); i++){
                        //distance_to_compute[i] = rot_first_half[i];
                        distance_to_compute.push_back(rot_first_half[rotation][i]);
                        //if(i == 1)
                        //   printf("rotation %d %d %lf\n",rotation, i, rot_first_half[rotation][i].position.x);
                    }
                    //cout << "main line " << __LINE__ << endl;
                    for(atom_st at : atoms_second_half){
                        //distance_to_compute[i] = at;
                        //i++;
                        distance_to_compute.push_back(at);
                    }
                }

                //cout << "main line " << __LINE__ << endl;
                    //cout << "main line " << __LINE__ << endl;
                    //int i = atoms_first_half.size();
                    /*
                    for(atom_st at : atoms_second_half){
                        //distance_to_compute[i] = at;
                        //i++;
                        //distance_to_compute.push_back(at);
                    }
                    cout << "main line " << __LINE__ << endl;
                    for(atom_st at : distance_to_compute){
                        cout <<at.id <<" "<< at.position.x << " " << at.position.y << " "<< at.position.z << endl;
                    }
                    */
                    //double res;
                    //printf("13 %lf x 33 %lf x\n", distance_to_compute[1].position.x, distance_to_compute[21].position.x);
                    res = distance_v3(distance_to_compute, atoms.size(), NUM_OF_STREAMS);
                    //printf("%d position pp %lf x %lf y %lf z\n", __LINE__, tmp.x,tmp.y,tmp.z);
                    //printf("%d prova print res1 %lf\n", __LINE__, res);
                    //cout << "main line " << __LINE__ << endl;

                    /*if(res > max_first_half.distance){
                        max_first_half.distance = res;
                        max_first_half.angle = c;
                        max_first_half.rot_mol_fst_half = rot_first_half[rotation];
                        max_first_half.rt = rt;
                    }*/
                    
                    for(int i = 0; i < NUM_OF_STREAMS;i++){
                        //printf("res[%d] %lf\n",i+c,res[i]);
                        if(res[i] > max_first_half.distance) {
                            max_first_half.distance = res[i];
                            max_first_half.angle = c+i;
                            max_first_half.rt = rt;
                        }
                    }
                    distance_to_compute.clear();
                    
                rot_first_half.clear();
            }


            printf("Computed distance for the first part,\n");
            printf("the max distance compute is %lf with angle %d around rotamer: %d\n", \
                    max_first_half.distance, max_first_half.angle,max_first_half.rt.getBond().getIdx());
            
            cudaFree(unit_quaternions);
        }
        else{
            printf("Checking rotamer %d ... ", rt.getBond().getIdx());
            printf("Too few atoms in the partition, rotamer not analized\n");
        }

        double total = max_first_half.distance + max_second_half.distance;

        if(total > max_dist.distance){
            max_dist.distance = total;
            max_dist.rt = max_first_half.rt;
            max_dist.angle = max_first_half.angle;
        }

        first_half.clear();
        second_half.clear();
        atoms_first_half.clear();
        atoms_second_half.clear();

        graph.addEdge(rt.getBeginAtom().id,rt.getEndingAtom().id);
        printf("For Rotamer %d, the max distance computed is: %lf,\n with a first angle: %d second angle: %d\n",\
                 rt.getBond().getIdx(),total,max_first_half.angle,max_second_half.angle);

        //sleep in order to let the cpu clean the memory to avoid random values
        //sleep(1);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

    cout << "duration time[ms]: " << duration.count() << endl;

    printf("The maximum distance computed is %lf\n", max_dist.distance);
    //int degree_first = max_.angle*(180/CUDART_PI);
    printf("Computed with an angle of %d, around the rotamer %d\n",max_dist.angle,max_dist.rt.getBond().getIdx());
    //cout << "Rotamer #0 startingNode: " << rotamers[0].getBeginAtom().id << " endingNode: " << rotamers[0].getEndingAtom().id << endl;
    //graph.removeEdge(rotamers[0].getBeginAtom().id,rotamers[0].getEndingAtom().id);
    //graph.to_string();
    /*
    vector<atom_st> atoms_first_half;
    vector<atom_st> atoms_second_half;
    for(auto i: first_half){
        atoms_first_half.push_back(atoms[i]);
    }
    for(auto i: second_half) atoms_second_half.push_back(atoms[i]);

    //Quaternion q = Quaternion(rotamers[0].getVector());
    float3 tmp = make_float3(0,0,2);
    Rotation r = Rotation(tmp);
    /*for(int i = 0; i < 360; i++){
        float radians = i * (CUDART_PI_F / 180);

    }

    atom_st* res_first_half;
    //float radians = degree * pi/180;
    /*
    for(int i = 0; i < first_half.size(); i++){
        cout << atoms_first_half[i].id << " " << atoms_first_half[i].position.x << " " << atoms_first_half[i].position.y << " " << atoms_first_half[i].position.z << endl;
    }

    res_first_half = r.rotate(CUDART_PI_F / 2 , atoms_first_half, tmp);

    atom_st distance_to_compute[atoms.size()];

    for(int i = 0; i < atoms_first_half.size(); i++){
        distance_to_compute[i] = res_first_half[i];
    }
    int i = atoms_first_half.size();
    for(atom_st at : atoms_second_half){
        distance_to_compute[i] = at;
        i++;
    }

    /*
    for(int i = 0; i < atoms.size(); i++){
        cout << distance_to_compute[i].id << " positions: " << distance_to_compute[i].position.x << " " << distance_to_compute[i].position.y << " "\
        << distance_to_compute[i].position.z << endl;
    }


    distance(distance_to_compute,atoms.size());

    /*
    std::cout << "Rotamers \n";
    for(Rotamer &rt : rotamers){
        int i = 0;
        RDKit::Bond bond = rt.getBond();
        std::cout << "Rot # " << i << bond.getBeginAtomIdx() << ' ' << bond.getEndAtomIdx() << std::endl;
        i++;
    }
    cout << rotamers[0].getBeginAtom().position.x << " " << rotamers[0].getBeginAtom().position.y << " " <<rotamers[0].getBeginAtom().position.z << "\n";
    cout << rotamers[0].getVector().x << " " << rotamers[0].getVector().y << " " << rotamers[0].getVector().z << "\n";

    */
    //sleep in order to let the gpu clean the memery before another run
    //sleep(2);
    return 0;

}
