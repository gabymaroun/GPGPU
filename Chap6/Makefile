NVCC=nvcc
CXX=nvc++
LIBCV=`pkg-config --libs opencv`
exe= 2-imgToGrayScale

all: $(exe)

%.o: %.cxx %.hpp
	$(CXX) -fPIC -c $(CFLAGS) $< -o $@
%.o: %.cxx
	$(CXX) -fPIC -c $(CFLAGS) $< -o $@
%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@

2-imgToGrayScale: 2-imgToGrayScale.o img_utils.o
	$(NVCC) $(CFLAGS) $^ -o $@  $(LDFLAGS) $(LIBCV) -lm



clean:
	rm -f *.o $(exe)
