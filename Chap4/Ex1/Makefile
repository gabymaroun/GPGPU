CXX=g++
NVCC=nvcc --std=c++11
CFLAGS=-O2

exe=1-basicMatMul

all: $(exe)

%.o: %.cxx %.hpp
	$(CXX) -c $(CFLAGS) $< -o $@
%.o: %.cxx
	$(CXX) -c $(CFLAGS) $< -o $@
%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@
1-basicMatMul: 1-basicMatMul.o matmul_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm
2-tiledMatMul: 2-tiledMatMul.o matmul_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm

clean:
	rm -f *.o $(exe)
