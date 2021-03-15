#ifndef ATOM_H_
#define ATOM_H_ 
#include<CL/sycl.hpp>
/**
+ * Struct used to define the structure of each atom keeping only the 
+ * useful information: id and position.
+ **/
struct atom_st {
    unsigned int atom_id;
    cl::sycl::double3 position;
};

#endif
