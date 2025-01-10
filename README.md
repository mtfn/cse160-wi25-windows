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
1. Ensure you have recent GPU drivers installed. You probably already have this. Note that you do not need any kind of CUDA toolkit because your drivers come with an OpenCL runtime.

2. Get a Bash port for Windows. I use Git Bash, which comes with Git for Windows. You can download it [here](https://git-scm.com/downloads). You'll also get a bunch of helpful Linux tools that are used by the Makefiles. This is technically optional but I tried using Powershell and it was a massive pain in the ass. Plus it's good to have Git anyway if you're gonna be coding on Windows.

3. Install Make and GCC. For me the easiest way to do this was through Scoop, a package manager for Windows. Install [Scoop](https://scoop.sh) and then run `scoop install gcc make` to install GCC and Make.

Now you should be able to use Git Bash to run the Makefiles and compile code as normal.

## Notes
If, when running your compiled code, you find that it stops midway through or you get an error message like `Error -1073741571`, you might be running into a stack overflow.
You can increase the stack size by editing compiler flags in Makefile:
```makefile 
CFLAGS = -O2 -std=c99 -Wl,--stack,268435456
```
You might notice I've already done this in a few places. 268435456 bytes might be overkill, you want to set it high enough that your code runs but not high enough that there any surprises when you submit to Gradescope. Remember that all submissions go through DSMLP so it really matters that it works there.