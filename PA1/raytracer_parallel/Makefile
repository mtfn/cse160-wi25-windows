CC = gcc 
CFLAGS = -O2 -Wall -Wl,--stack,268435456

LDFLAGS += -L../../OpenCL-SDK/lib -lOpenCL
INCFLAGS += -I../../OpenCL-SDK/include
MATHFLAG = -lm

all: raytracer_parallel
raytracer_parallel: main.c lib/vec_ops.c 
	$(CC) $(CFLAGS) -o raytracer_parallel main.c $(INCFLAGS) $(LDFLAGS) $(MATHFLAG)

gpu: raytracer_parallel
	./raytracer_parallel gpu

cpu: raytracer_parallel
	./raytracer_parallel cpu

clean:
	rm -f raytracer_parallel