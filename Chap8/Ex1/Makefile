NVCC=nvcc
CXX=nvc++
LIBCV=`pkg-config --libs opencv`
exe= 0-convolutionCPU

all: $(exe)

%.o: %.cxx %.hpp
	$(CXX) -fPIC -c $(CFLAGS) $< -o $@
%.o: %.cxx
	$(NVCC) -c $(CFLAGS) $< -o $@
%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@
0-convolutionCPU: 0-convolutionCPU.o img_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm

clean:
	rm -f *.o $(exe)
