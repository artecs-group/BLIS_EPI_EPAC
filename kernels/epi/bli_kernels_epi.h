/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

GEMM_UKR_PROT( float,   s, gemm_epi_scalar_24x8 )
GEMM_UKR_PROT( float,   s, gemm_epi_scalar_8x3v )
GEMM_UKR_PROT( double,   d, gemm_epi_scalar_16x1v )
GEMM_UKR_PROT( double,   d, gemm_epi_scalar_8x3v )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_epi_int_8x3vm )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_epi_int_8x3vn )
GEMMSUP_KER_PROT( double,    d, gemmsup_rv_epi_int_8x3vm )

//GEMMSUP_KER_PROT( double,    d, gemmsup_rv_haswell_asm_6x8 )

GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_l_epi_scalar_16x1v )
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_l_epi_scalar_8x3v )
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_u_epi_scalar_16x1v )
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_u_epi_scalar_8x3v )

AXPYV_KER_PROT( float,    s, axpyv_epi_int )
AXPYV_KER_PROT( double,   d, axpyv_epi_int )

AMAXV_KER_PROT( float,    s, amaxv_epi_int )
AMAXV_KER_PROT( double,   d, amaxv_epi_int )

SCALV_KER_PROT( float,    s, scalv_epi_int )
SCALV_KER_PROT( double,   d, scalv_epi_int )

SWAPV_KER_PROT(float,    s, swapv_epi_int )
SWAPV_KER_PROT(double,   d, swapv_epi_int )

COPYV_KER_PROT( float,    s, copyv_epi_int )
COPYV_KER_PROT( double,   d, copyv_epi_int )

SETV_KER_PROT(float,    s, setv_epi_int)
SETV_KER_PROT(double,   d, setv_epi_int)



