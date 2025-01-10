CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

LDFLAGS += -L../../OpenCL-SDK/lib -lOpenCL
INCFLAGS += -I../../OpenCL-SDK/include

all: device_query

device_query: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

run: device_query
	@./device_query

clean: 
	@rm -f device_query