CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS :=

LDFLAGS += -lm
LDFLAGS += -L../OpenCL-SDK/lib -lOpenCL
INCFLAGS += -I../OpenCL-SDK/include

SOURCES := device.c kernel.c matrix.c img.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all
.SUFFIXES: .o .c
all: helper_lib.a

debug: CFLAGS += -DOCL_DEVICE_TYPE=CL_DEVICE_TYPE_CPU
debug: helper_lib.a

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $^ $(INCFLAGS) $(LDFLAGS)

helper_lib.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)
clean: 
	rm -r $(OBJECTS) helper_lib.a
