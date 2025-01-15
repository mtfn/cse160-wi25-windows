CC       = gcc
CFLAGS   = -g -Wall -Wl,--stack,268435456
INCFLAGS := -I../helper_lib
LDFLAGS  := ../helper_lib/helper_lib.a -lm

LDFLAGS += -L../../OpenCL-SDK/lib -lOpenCL
INCFLAGS += -I../../OpenCL-SDK/include

all: solution

solution: ../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../helper_lib/helper_lib.a: 
	cd ../helper_lib; make

run: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/input2.raw Dataset/0/input3.raw Dataset/0/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/input2.raw Dataset/1/input3.raw Dataset/1/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/input2.raw Dataset/2/input3.raw Dataset/2/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/input2.raw Dataset/3/input3.raw Dataset/3/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/input2.raw Dataset/4/input3.raw Dataset/4/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/input2.raw Dataset/5/input3.raw Dataset/5/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/input2.raw Dataset/6/input3.raw Dataset/6/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/input2.raw Dataset/7/input3.raw Dataset/7/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/8/input0.raw Dataset/8/input1.raw Dataset/8/input2.raw Dataset/8/input3.raw Dataset/8/output.raw program_1_output.raw program_2_output.raw
	./solution Dataset/9/input0.raw Dataset/9/input1.raw Dataset/9/input2.raw Dataset/9/input3.raw Dataset/9/output.raw program_1_output.raw program_2_output.raw

clean: 
	rm -f parallel sequential
	cd ../helper_lib; make clean
