/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.
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

void bli_scopyv_epi_int
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t*          cntx
     )
{
	const uint32_t num_elem_per_reg = 8;
	uint32_t       i = 0;
	__epi_2xf32      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, xv11, xv12, xv13, xv14, xv15;

	long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 && incy == 1 )
	{

		// For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
		// for example if n = 255
		// n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
		// n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
		// n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
		for ( i = 0; i < (n & (~0x7F)); i += 128 )
		{
			xv0 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 3, gvl);
			xv4 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 4, gvl);
			xv5 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 5, gvl);
			xv6 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 6, gvl);
			xv7 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 7, gvl);
			xv8 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 8, gvl);
			xv9 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 9, gvl);
			xv10 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 10, gvl);
			xv11 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 11, gvl);
			xv12 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 12, gvl);
			xv13 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 13, gvl);
			xv14 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 14, gvl);
			xv15 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 15, gvl);

			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 3, xv3, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 4, xv4, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 5, xv5, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 6, xv6, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 7, xv7, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 8, xv8, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 9, xv9, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 10, xv10, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 11, xv11, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 12, xv12, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 13, xv13, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 14, xv14, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 15, xv15, gvl);

			y += 128;
			x += 128;
		}
		for ( ; i < (n & (~0x3F)); i += 64 )
		{
			xv0 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 3, gvl);
			xv4 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 4, gvl);
			xv5 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 5, gvl);
			xv6 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 6, gvl);
			xv7 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 7, gvl);

			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 3, xv3, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 4, xv4, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 5, xv5, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 6, xv6, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 7, xv7, gvl);

			y += 64;
			x += 64;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			xv0 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 3, gvl);

			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 3, xv3, gvl);

			y += 32;
			x += 32;
		}
		for ( ; i < (n & (~0x0F)); i += 16 )
		{
			xv0 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 1, gvl);

			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 1, xv1, gvl);

			y += 16;
			x += 16;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			xv0 = __builtin_epi_vload_2xf32(x + num_elem_per_reg * 0, gvl);
			__builtin_epi_vstore_2xf32(y + num_elem_per_reg * 0, xv0, gvl);
			y += 8;
			x += 8;
		}
		for ( ; i < n; ++i )
		{
			*y++ = *x++;
		}
	}
	else
	{
		for ( dim_t i = 0; i < n; ++i )
		{
			*y = *x;
			x += incx;
			y += incy;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_dcopyv_epi_int
     (
       conj_t           conjx,
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t*          cntx
     )
{
	const uint32_t num_elem_per_reg = 4;
	uint32_t       i = 0;
	__epi_1xf64      xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xv8, xv9, xv10, xv11, xv12, xv13, xv14, xv15;

	long gvl = __builtin_epi_vsetvl( 4, __epi_e64, __epi_m1 );

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 && incy == 1 )
	{
		// n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
		// the copy operation will be done for the multiples of 64
		for ( i = 0; i < (n & (~0x3F)); i += 64 )
		{
			xv0 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 3, gvl);
			xv4 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 4, gvl);
			xv5 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 5, gvl);
			xv6 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 6, gvl);
			xv7 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 7, gvl);
			xv8 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 8, gvl);
			xv9 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 9, gvl);
			xv10 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 10, gvl);
			xv11 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 11, gvl);
			xv12 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 12, gvl);
			xv13 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 13, gvl);
			xv14 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 14, gvl);
			xv15 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 15, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 3, xv3, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 4, xv4, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 5, xv5, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 6, xv6, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 7, xv7, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 8, xv8, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 9, xv9, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 10, xv10, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 11, xv11, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 12, xv12, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 13, xv13, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 14, xv14, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 15, xv15, gvl);
			y += num_elem_per_reg * 16;
			x += num_elem_per_reg * 16;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			xv0 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 3, gvl);
			xv4 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 4, gvl);
			xv5 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 5, gvl);
			xv6 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 6, gvl);
			xv7 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 7, gvl);

			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 3, xv3, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 4, xv4, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 5, xv5, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 6, xv6, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 7, xv7, gvl);

			y += num_elem_per_reg * 8;
			x += num_elem_per_reg * 8;
		}
		for ( ; i < (n & (~0xF)); i += 16 )
		{
			xv0 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 1, gvl);
			xv2 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 2, gvl);
			xv3 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 3, gvl);

			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 1, xv1, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 2, xv2, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 3, xv3, gvl);

			y += num_elem_per_reg * 4;
			x += num_elem_per_reg * 4;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			xv0 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 0, gvl);
			xv1 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 1, gvl);

			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 0, xv0, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 1, xv1, gvl);

			y += num_elem_per_reg * 2;
			x += num_elem_per_reg * 2;
		}
		for ( ; i < (n & (~0x03)); i += 4 )
		{
			xv0 = __builtin_epi_vload_1xf64(x + num_elem_per_reg * 0, gvl);
			__builtin_epi_vstore_1xf64(y + num_elem_per_reg * 0, xv0, gvl);
			y += num_elem_per_reg;
			x += num_elem_per_reg;
		}
		for ( ; i < n; ++i )
		{
			*y++ = *x++;
		}
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			*y = *x;

			x += incx;
			y += incy;
		}
	}
}
