NVCC=nvcc
CXX=nvc++
LIBCV=`pkg-config --libs opencv`
exe= 2-tiledConvolutionCPU

all: $(exe)

%.o: %.cxx %.hpp
	$(CXX) -fPIC -c $(CFLAGS) $< -o $@
%.o: %.cxx
	$(NVCC) -c $(CFLAGS) $< -o $@
%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@
2-tiledConvolutionCPU: 2-tiledConvolutionCPU.o img_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm

clean:
	rm -f *.o $(exe)
