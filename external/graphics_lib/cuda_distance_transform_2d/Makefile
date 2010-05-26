################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= voronoi2d
# CUDA source files (compiled with cudacc)
CUFILES		:= add_one_kernel.cu map.cu scan_efficient.cu scan_efficient_8.cu
# CUDA dependency files
CU_DEPS		:= 
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp  ping_pong_cuda.cpp distance_transform_2d.cpp globals.cpp timege.cpp centroids.cpp

################################################################################
GLH = ${GRAPHIC_LIBS_HOME}

GLINC  = \
	-I. \
	-I${LOCAL_CUDA}/common/inc \
	-I${GLH}/graphics_utils \
	-I${GLH}/utilities \
	-I${GLH}/cuda_utilities \
	-I${GLH}/random

GLLIBS = \
	-L${GLH}/graphics_utils \
	-L${GLH}/random -L${GLH}/utilities \
	-L${GLH}/cuda_utilities \
	-L/usr/lib64 \
	-L${GLH}/utilities_runtime 

LIB :=  \
	${GLLIBS} \
	-lutilities  \
	-lcuda_utilities  \
	-lgraphic_utilities \
	-lrandom -L/usr/lib \
	-lcuda \
	-lGLEW

################################################################################
# Rules and targets

# create an environment variable LOCAL_CUDA for use in your Makefiles
#
# used in common.mk
ROOTDIR := ${LOCAL_CUDA}/projects

#emu=1
dbg=1
ROOTBINDIR = .
INCLUDES := ${GLINC}

include ${LOCAL_CUDA}/common/common.mk


