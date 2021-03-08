#ifndef ROTAMER_H_
#define ROTAMER_H_


#include <GraphMol/GraphMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include "rotation_cuda.cuh"
#include "atom_cuda.cuh"

class Rotamer{

public:
    
    Rotamer()=default;
    
    Rotamer(const RDKit::Bond b,
            unsigned int id,
            atom_st &beginAtom,
            atom_st &endAtom);

    RDKit::Bond getBond();

    atom_st getBeginAtom();
    atom_st getEndingAtom();
    double3 getVector();

private:
    RDKit::Bond bond;
    atom_st beginAtom;
    atom_st endAtom;
    unsigned int idx;
    double3 vector;
    //Quaternion quaternion;

    double3 ComputeVector(atom_st &startingAtom, atom_st &endAtom);
    
};
#endif //ROTAMER_H_