NAME	= PPproject
CC	= nvcc
CFLAGS	= -I${RDBASE}/Code -L${RDBASE}/Extern -I${BOOST_ROOT} -L${RDBASE}/lib -I ${CUDA_BASE}/include -L ${CUDA_BASE}/lib -std=c++11 -O3
LDFLAGS	= -lRDKitChemReactions -lRDKitFileParsers -lRDKitSmilesParse -lRDKitDepictor  -lRDKitSubstructMatch -lRDKitGraphMol -lRDKitDataStructs -lRDKitRDGeometryLib -lRDKitRDGeneral 
SRCDIR	= src
CUDA_ARCH = 61
ARCH = -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} -arch sm_${CUDA_ARCH}
CUDAFLAGS = -lcudadevrt  -rdc=true 
all:
	$(CC) -g  ${ARCH}  $(SRCDIR)/*.cu $(SRCDIR)/*.cpp  -o main $(CFLAGS) $(LDFLAGS) $(CUDAFLAGS)
