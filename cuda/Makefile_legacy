NAME	= Cuda-Docking
CC	= nvcc
CFLAGS	= -ccbin=g++-11 -I${RDBASE}/Code -L${RDBASE}/External -I${BOOST_ROOT}/include -L${BOOST_ROOT}/lib -L${RDBASE}/lib -I ${CUDA_BASE_DIR}/include -L ${CUDA_BASE_DIR}/lib64 -std=c++11 -O3 -I /src/
LDFLAGS	= -lRDKitChemReactions -lRDKitFileParsers -lRDKitSmilesParse -lRDKitDepictor  -lRDKitSubstructMatch -lRDKitGraphMol -lRDKitDataStructs -lRDKitRDGeometryLib -lRDKitRDGeneral 

OBJDIR = obj/
SRCDIR	= src

CUDA_ARCH = 61
ARCH = -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} -arch sm_${CUDA_ARCH}
CUDAFLAGS = -lcudadevrt  -rdc=true 

_DEPS = atom.cuh distance.h graph.h rotation.cuh rotamer.cuh helper.h utils.h
DEPS = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ = distance.o graph.o rotation.o rotamer.o main.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

$(OBJDIR)%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -g $(ARCH) -c -o $@ $< $(CFLAGS) $(CUDAFLAGS) $(LDFLAGS) 

$(OBJDIR)%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -g $(ARCH) -c -o $@ $< $(CFLAGS) $(CUDAFLAGS) $(LDFLAGS) 


main: $(OBJ)
	$(CC) -g  ${ARCH}  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(CUDAFLAGS)
