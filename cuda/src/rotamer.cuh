#ifndef ROTAMER_H_
#define ROTAMER_H_


#include <GraphMol/GraphMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>



#include "rotation.cuh"
#include "atom.cuh"

class Rotamer{

public:
    
    Rotamer()=default;
    
    Rotamer(const RDKit::Bond b,
            unsigned int id,
            atom_st &beginAtom,
            atom_st &endAtom);

    RDKit::Bond getBond() const ;

    atom_st getBeginAtom() const;
    atom_st getEndingAtom() const;
    double3 getVector() const;

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