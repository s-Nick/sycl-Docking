#ifndef UTILS_H_
#define UTILS_H_

#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/FileParsers.h>

#include<fstream>
#include<vector>
#include<string>

void multipleMoleculeRead(std::ifstream& molFileStream, 
                    std::vector<std::shared_ptr<RDKit::ROMol>>& molecules){
    
    std::string line;
    std::stringstream ss;
    while(!molFileStream.eof()){
        std::fstream molStream;
        molStream.open("tmpdata.mol2", std::ios::out);
        getline(molFileStream, line);
        if(line == "@<TRIPOS>MOLECULE"){
            //std::cout << line << " at line " << __LINE__ << std::endl;
            std::streampos mol_init = molFileStream.tellg();
            ss << line;
            ss <<'\n'; 
            //molStream << ss.rdbuf();
            //std::cout << molStream.rdbuf() << __LINE__ << std::endl;
            //std::cout << ss.rdbuf() << " at line " << __LINE__ << std::endl;
            getline(molFileStream, line);
            while(line != "@<TRIPOS>MOLECULE" && !molFileStream.eof()){
                ss << line;
                //molStream << ss.rdbuf();
                ss << '\n';
                getline(molFileStream, line);
            }
            molStream << ss.rdbuf();
            molStream.close();
            std::fstream molStream;
            molStream.open("tmpdata.mol2", std::ios::in);
            //std::cout << molStream.rdbuf() << std::endl;
            std::shared_ptr<RDKit::ROMol> const mol(RDKit::Mol2DataStreamToMol(molStream, false,true, RDKit::CORINA, false));
            molecules.push_back(mol);
            if(line == "\0")
                break;
            else
                molFileStream.seekg(mol_init);
            //std::cout << ss.rdbuf() << std::endl;
            ss.clear();
            molStream.close();
        }
    }
    return;
}

void singleMoleculeRead(std::ifstream& molFileStream, 
                    std::vector<std::shared_ptr<RDKit::ROMol>>& molecules){
    
    std::string line;
    std::stringstream ss;
    while(!molFileStream.eof()){
        std::fstream molStream;
        molStream.open("tmpdata.mol2", std::ios::out);
        getline(molFileStream, line);
        //std::cout << line << std::endl;
        if(line.compare("@<TRIPOS>MOLECULE")){
            //std::cout << line << " at line " << __LINE__ << std::endl;
            std::streampos mol_init = molFileStream.tellg();
            ss << line;
            ss <<'\n'; 
            //molStream << ss.rdbuf();
            //std::cout << molStream.rdbuf() << __LINE__ << std::endl;
            //std::cout << ss.rdbuf() << " at line " << __LINE__ << std::endl;
            getline(molFileStream, line);
            while(line != "@<TRIPOS>MOLECULE" && !molFileStream.eof()){
                ss << line;
                //molStream << ss.rdbuf();
                ss << '\n';
                //std::cout <<__LINE__<<std::endl;
                getline(molFileStream, line);
            }
            molStream << ss.rdbuf();
            molStream.close();
            std::fstream molStream;
            molStream.open("tmpdata.mol2", std::ios::in);
            //std::cout << molStream.rdbuf() << std::endl;
            std::shared_ptr<RDKit::ROMol> const mol(RDKit::Mol2DataStreamToMol(molStream, false,true, RDKit::CORINA, false));
            molecules.push_back(mol);
            if(line == "\0")
                break;
            else
                molFileStream.seekg(mol_init);
            //std::cout << ss.rdbuf() << std::endl;
            ss.clear();
            molStream.close();
        }
    }
    //molStream.close();
    //auto tmp = molecules[3]->getProp<std::string>("_Name");
    //auto s = mol->getProp<std::string>("_Name");
    auto tmp = molecules.size();
    std::cout << tmp << std::endl;


    return;
}


#endif
