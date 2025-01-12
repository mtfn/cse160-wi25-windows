# cse160-opencl
---
CSE 160 Programming Assignments using OpenCL

## Instruction
---
Refer to the CSE160 Documentation -> [Documentation](https://docs-cse160.readthedocs.io/en/latest/)

---

This is the local setup I've been using on my Windows PC with an Nvidia GPU (not tested on AMD yet, but I have an AMD CPU with integrated graphics and it seems to work with that).

The bulk of the changes I've made are just editing the Makefiles to link the OpenCL SDK, which I've also included in this repo.

## Setup
You will need:
- A Bash port for Windows. I use Git Bash, which comes with Git for Windows. You might have this from a previous class. If not, you can download it [here](https://git-scm.com/downloads). It comes with a bunch of helpful Linux utilities that the Makefiles will use.
- GCC and Make. For me the easiest way to install them was through Scoop, a package manager for Windows. [Install Scoop](https://scoop.sh) if you haven't and then run `scoop install gcc make`.
- Recent GPU drivers, of course. Windows probably took care of this already.

Now you should be able to use Git Bash to run the Makefiles and compile code as normal.

## Notes

### Stack overflow
If, when running your compiled code, you find that it stops midway through or you get an error message like `Error -1073741571`, you might be running into a stack overflow.
You can increase the stack size by editing compiler flags in Makefile and then recompiling.
```makefile
CFLAGS = -O2 -std=c99 -Wl,--stack,268435456
```
You might notice I've already done this in a few places. 268435456 bytes might be overkill; you want to set it high enough that your code runs locally but not high enough that there any surprises when you submit to Gradescope.

### Platform number
The programming assignments all assume that the GPU is platform 0 device 0. If that is not the case for your PC, you might have to go into the code for `main.c`
```c
device_id = platforms[0].devices[0].device_id;
```
and change it accordingly, then **change it back when you submit**. Note that `export POCL_DEVICES=cuda` doesn't apply to this setup so don't bother with that.