#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#


# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := epi
CONFIGS_INCL   += $(THIS_CONFIG)

#
# --- Determine the C compiler and related flags ---
#

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.
CPPROCFLAGS    := -D_GNU_SOURCE #-O2 -mepi #-fno-vectorize -D_GNU_SOURCE #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
CMISCFLAGS     := -O2 -mepi -fno-vectorize -mcpu=avispado #-O2 -std=gnu11 #-D_POSIX_C_SOURCE=200809L
CPICFLAGS      := #-O2 -mepi #-fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
CWARNFLAGS     := #-O2 -mepi #-fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g #-O2 -mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := #-O0 -mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
else
COPTFLAGS      := #-O2 -mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
endif

# Flags specific to optimized kernels.
CKOPTFLAGS     := $(COPTFLAGS) #-O3
ifeq ($(CC_VENDOR),clang)
CKVECFLAGS     := -O2 -mepi -fno-vectorize -mcpu=avispado #-mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
else
$(error clang is required for this configuration.)
endif

# Flags specific to reference kernels.
CROPTFLAGS     := $(CKOPTFLAGS) #-mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
ifeq ($(CC_VENDOR),clang)
#CRVECFLAGS     := $(CKVECFLAGS) -fopenmp-simd #-mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
CRVECFLAGS     := -fopenmp-simd  -Rpass=loop-vectorize #-Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize #-mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
else
CRVECFLAGS     := $(CKVECFLAGS) #-mepi -fno-vectorize #-std=gnu11 #-D_POSIX_C_SOURCE=200809L
endif

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))

