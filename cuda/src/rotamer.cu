#include "rotamer.cuh"
#include <GraphMol/GraphMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include "atom.cuh"


/**
 * Constructor of the Rotamer
 * It build the vector that characterize the bond
 * 
 * @param b the bond that it construct the rotamer,
 * @param  id the id of the bond in the molecule,
 * @param  bAtom the beggining atom of the bond,
 * @param eAtom the atom at the end of the bond 
 **/
Rotamer::Rotamer(const RDKit::Bond b,
            unsigned int id,
            atom_st &bAtom,
            atom_st &eAtom):bond(b),beginAtom(bAtom),endAtom(eAtom){
    idx = id;
    vector = ComputeVector(beginAtom,endAtom);
}

/**
 * Compute the vector of the corresponding bond, starting from the atoms at the begin and at the end of it.
 * The coordinates of the vector are computed as the difference of the atoms.
 * 
 * @param startingAtom the atom at the beginning of the bond.
 * @param endAtom the atom at the end of the bond
 * 
 **/
double3 Rotamer::ComputeVector(atom_st &startingAtom, atom_st &endAtom){
    double3 vector;
    vector.x = endAtom.position.x - startingAtom.position.x;
    vector.y = endAtom.position.y - startingAtom.position.y;
    vector.z = endAtom.position.z - startingAtom.position.z;
    return vector;
}


/**
 * Return the bond corresponding to this rotamer.
 **/
RDKit::Bond Rotamer::getBond(){
    return bond;
}

/**
 * Return the atom at the begin of the rotamer.
 **/ 
atom_st Rotamer::getBeginAtom(){
    return beginAtom;
}

/**
 * Return the atom at the end of the rotamer.
 **/
atom_st Rotamer::getEndingAtom(){
    return endAtom;
}

/**
 * Return the computed vector of the corresponding bond. 
 **/
double3 Rotamer::getVector(){
    return vector;
}
