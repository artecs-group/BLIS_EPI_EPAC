/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   Copyright (C) 2020, Advanced Micro Devices, Inc.
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

#include "blis.h"

// -----------------------------------------------------------------------------

void bli_ssetv_epi_int
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       cntx_t*          cntx
     )
{
	const uint32_t num_elem_per_reg = 8;
	uint32_t       i = 0;
	__epi_2xf32      alphav;

	long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 )
	{
	
		alphav = __builtin_epi_vfmv_v_f_2xf32( *alpha, gvl );

		// For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
		// for example if n = 255
		// n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
		// n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
		// n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
		for ( i = 0; i < (n & (~0x7F)); i += 128 )
		{
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 3, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 4, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 5, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 6, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 7, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 8, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 9, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 10, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 11, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 12, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 13, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 14, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 15, alphav, gvl);

			x += 128;
		}
		for ( ; i < (n & (~0x3F)); i += 64 )
		{
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 3, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 4, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 5, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 6, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 7, alphav, gvl);

			x += 64;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 3, alphav, gvl);

			x += 32;
		}
		for ( ; i < (n & (~0x0F)); i += 16 )
		{
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 1, alphav, gvl);

			x += 16;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			__builtin_epi_vstore_2xf32(x + num_elem_per_reg * 0, alphav, gvl);
			x += 8;
		}
		for ( ; i < n; ++i )
		{
			*x++ = *alpha;
		}
	}
	else
	{
		for ( dim_t i = 0; i < n; ++i )
		{
			*x = *alpha;
			x += incx;
		}
	}
}

void  bli_dsetv_epi_int
     (
       conj_t           conjalpha,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       cntx_t*          cntx
     )
{
	const uint32_t num_elem_per_reg = 4;
	uint32_t       i = 0;
	__epi_1xf64     alphav;

	long gvl = __builtin_epi_vsetvl( 4, __epi_e64, __epi_m1 );

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = __builtin_epi_vfmv_v_f_1xf64( *alpha, gvl );

		// n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
		// the copy operation will be done for the multiples of 64
		for ( i = 0; i < (n & (~0x3F)); i += 64 )
		{
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 3, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 4, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 5, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 6, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 7, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 8, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 9, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 10, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 11, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 12, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 13, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 14, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 15, alphav, gvl);

			x += num_elem_per_reg * 16;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 3, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 4, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 5, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 6, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 7, alphav, gvl);

			x += num_elem_per_reg * 8;
		}
		for ( ; i < (n & (~0xF)); i += 16 )
		{
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 1, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 2, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 3, alphav, gvl);

			x += num_elem_per_reg * 4;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 0, alphav, gvl);
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 1, alphav, gvl);

			x += num_elem_per_reg * 2;
		}
		for ( ; i < (n & (~0x03)); i += 4 )
		{
			__builtin_epi_vstore_1xf64(x + num_elem_per_reg * 0, alphav, gvl);
			x += num_elem_per_reg;
		}
		for ( ; i < n; ++i )
		{
			*x++ = *alpha;
		}
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			*x = *alpha;

			x += incx;
		}
	}
}

