CXX=nvc++ -acc=gpu -Minfo=accel
NVCC=nvc++ -acc=gpu -Minfo=accel
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

2-basicMatMul: 2-basicMatMul.o matmul_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm

3-basicMatMul: 3-basicMatMul.o matmul_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm

clean:
	rm -f *.o $(exe)
