CC       = gcc
CFLAGS   = -g -Wall -Wl,--stack,268435456
INCFLAGS := -I../helper_lib
LDFLAGS  := ../helper_lib/helper_lib.a -lm
MAKECMD  = make

LDFLAGS += -L../OpenCL-SDK/lib -lOpenCL
INCFLAGS += -I../OpenCL-SDK/include

all: solution

debug: CFLAGS += -DOCL_DEVICE_TYPE=CL_DEVICE_TYPE_CPU
debug: MAKECMD += debug
debug: solution

solution: ../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../helper_lib/helper_lib.a: 
	cd ../helper_lib; $(MAKECMD)

run: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/output.raw output.raw
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/output.raw output.raw
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output.raw output.raw
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/output.raw output.raw
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/output.raw output.raw
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/output.raw output.raw
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/output.raw output.raw
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/output.raw output.raw
	./solution Dataset/8/input0.raw Dataset/8/input1.raw Dataset/8/output.raw output.raw
	./solution Dataset/9/input0.raw Dataset/9/input1.raw Dataset/9/output.raw output.raw
	./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw

time: solution
	python3 ../utils/profile.py --args ./solution Dataset/10/input0.raw Dataset/10/input1.raw Dataset/10/output.raw output.raw
	
clean: 
	rm -f solution
	cd ../helper_lib; make clean
