CC = gcc 
CFLAGS = -O2 -Wall

ifeq ($(shell uname -s),Darwin)
    LDFLAGS = -framework OpenCL
else
    LDFLAGS = -L/usr/local/cuda/lib64 -lOpenCL
endif

# might not be necessary, uncomment if you need it
# INCFLAGS = -I/usr/local/cuda/include
MATHFLAG = -lm

all: draw
draw: main.c
	$(CC) $(CFLAGS) -o draw main.c $(INCFLAGS) $(LDFLAGS) $(MATHFLAG) -w

clean:
	rm -f draw
