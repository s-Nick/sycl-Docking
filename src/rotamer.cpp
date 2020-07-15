#include "rotamer.cuh"
#include <GraphMol/GraphMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include "atom.cuh"

Rotamer::Rotamer(const RDKit::Bond b,
            unsigned int id,
            atom_st &bAtom,
            atom_st &eAtom):bond(b),beginAtom(bAtom),endAtom(eAtom){
    idx = id;
    vector = ComputeVector(beginAtom,endAtom);
}

float3 Rotamer::ComputeVector(atom_st &startingAtom, atom_st &endAtom){
    float3 vector;
    vector.x = endAtom.position.x - startingAtom.position.x;
    vector.y = endAtom.position.y - startingAtom.position.y;
    vector.z = endAtom.position.z - startingAtom.position.z;
    return vector;
}

RDKit::Bond Rotamer::getBond(){
    return bond;
}

atom_st Rotamer::getBeginAtom(){
    return beginAtom;
}
atom_st Rotamer::getEndingAtom(){
    return endAtom;
}
float3 Rotamer::getVector(){
    return vector;
}
