NAME	= proj
CC	= nvcc
CFLAGS	= -I${RDBASE}/Code -L${RDBASE}/Extern -I${BOOST_ROOT} -L${RDBASE}/lib -I /usr/local/cuda-10.2/targets/x86_64-linux/include -L /usr/local/cuda-10.2/targets/x86_64-linux/lib
LDFLAGS	= -lRDKitChemReactions -lRDKitFileParsers -lRDKitSmilesParse -lRDKitDepictor  -lRDKitSubstructMatch -lRDKitGraphMol -lRDKitDataStructs -lRDKitRDGeometryLib -lRDKitRDGeneral 
SRCDIR	= src
CUDA_ARCH = 50
ARCH = -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}
CUDAFLAGS = -lcudadevrt -lcublas -lcublas_static -rdc=true #-use_fast_math
all:
	$(CC) -g  ${ARCH}  $(SRCDIR)/*.cu $(SRCDIR)/*.cpp  -o main $(CFLAGS) $(LDFLAGS) $(CUDAFLAGS)
