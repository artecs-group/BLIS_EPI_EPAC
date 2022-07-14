/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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



void bli_saxpyv_epi_int
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const uint32_t      n_elem_per_reg = 8;
	const uint32_t      n_iter_unroll  = 4;

	uint32_t            i;
	uint32_t            n_viter;
	uint32_t            n_left;

	float*  restrict x0;
	float*  restrict y0;

	__epi_2xf32           alphav;
	__epi_2xf32           x0v, x1v, x2v, x3v;
	__epi_2xf32           y0v, y1v, y2v, y3v;

	long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );
	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Broadcast the alpha scalar to all elements of a vector register.
	//alphav.v = _mm256_broadcast_ss( alpha );
	alphav = __builtin_epi_vfmv_v_f_2xf32( *alpha, gvl );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < n_viter; ++i )
	{
		// Load the input values.
		y0v = __builtin_epi_vload_2xf32( y0 + 0*n_elem_per_reg, gvl );
		x0v = __builtin_epi_vload_2xf32( x0 + 0*n_elem_per_reg, gvl );

		y1v = __builtin_epi_vload_2xf32( y0 + 1*n_elem_per_reg, gvl );
		x1v = __builtin_epi_vload_2xf32( x0 + 1*n_elem_per_reg, gvl );

		y2v = __builtin_epi_vload_2xf32( y0 + 2*n_elem_per_reg, gvl );
		x2v = __builtin_epi_vload_2xf32( x0 + 2*n_elem_per_reg, gvl );

		y3v = __builtin_epi_vload_2xf32( y0 + 3*n_elem_per_reg, gvl );
		x3v = __builtin_epi_vload_2xf32( x0 + 3*n_elem_per_reg, gvl );

		// perform : y += alpha * x;
		y0v = __builtin_epi_vfmadd_2xf32( alphav, x0v, y0v, gvl );
		y1v = __builtin_epi_vfmadd_2xf32( alphav, x1v, y1v, gvl );
		y2v = __builtin_epi_vfmadd_2xf32( alphav, x2v, y2v, gvl );
		y3v = __builtin_epi_vfmadd_2xf32( alphav, x3v, y3v, gvl );

		// Store the output.
		__builtin_epi_vstore_2xf32( (y0 + 0*n_elem_per_reg), y0v, gvl );
		__builtin_epi_vstore_2xf32( (y0 + 1*n_elem_per_reg), y1v, gvl );
		__builtin_epi_vstore_2xf32( (y0 + 2*n_elem_per_reg), y2v, gvl );
		__builtin_epi_vstore_2xf32( (y0 + 3*n_elem_per_reg), y3v, gvl );

		x0 += n_elem_per_reg * n_iter_unroll;
		y0 += n_elem_per_reg * n_iter_unroll;
	}


	const float alphac = *alpha;

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const float x0c = *x0;

		*y0 += alphac * x0c;

		x0 += incx;
		y0 += incy;
	}
}

// -----------------------------------------------------------------------------

void bli_daxpyv_epi_int
     (
       conj_t           conjx,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const uint32_t       n_elem_per_reg = 4;
	const uint32_t       n_iter_unroll  = 4;

	uint32_t             i;
	uint32_t             n_viter;
	uint32_t             n_left;

	double*  restrict x0;
	double*  restrict y0;

	__epi_1xf64       alphav;
	__epi_1xf64       x0v, x1v, x2v, x3v;
	__epi_1xf64       y0v, y1v, y2v, y3v;
	
	long gvl = __builtin_epi_vsetvl( 4, __epi_e64, __epi_m1 );

	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Broadcast the alpha scalar to all elements of a vector register.
	//alphav.v = _mm256_broadcast_sd( alpha );
	alphav = __builtin_epi_vfmv_v_f_1xf64( *alpha, gvl );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < n_viter; ++i )
	{
		// Load the input values.
		y0v = __builtin_epi_vload_1xf64( y0 + 0*n_elem_per_reg, gvl );
		x0v = __builtin_epi_vload_1xf64( x0 + 0*n_elem_per_reg, gvl );

		y1v = __builtin_epi_vload_1xf64( y0 + 1*n_elem_per_reg, gvl );
		x1v = __builtin_epi_vload_1xf64( x0 + 1*n_elem_per_reg, gvl );

		y2v = __builtin_epi_vload_1xf64( y0 + 2*n_elem_per_reg, gvl );
		x2v = __builtin_epi_vload_1xf64( x0 + 2*n_elem_per_reg, gvl );

		y3v = __builtin_epi_vload_1xf64( y0 + 3*n_elem_per_reg, gvl );
		x3v = __builtin_epi_vload_1xf64( x0 + 3*n_elem_per_reg, gvl );

		// perform : y += alpha * x;
		y0v = __builtin_epi_vfmadd_1xf64( alphav, x0v, y0v, gvl );
		y1v = __builtin_epi_vfmadd_1xf64( alphav, x1v, y1v, gvl );
		y2v = __builtin_epi_vfmadd_1xf64( alphav, x2v, y2v, gvl );
		y3v = __builtin_epi_vfmadd_1xf64( alphav, x3v, y3v, gvl );

		// Store the output.
		__builtin_epi_vstore_1xf64( (y0 + 0*n_elem_per_reg), y0v, gvl );
		__builtin_epi_vstore_1xf64( (y0 + 1*n_elem_per_reg), y1v, gvl );
		__builtin_epi_vstore_1xf64( (y0 + 2*n_elem_per_reg), y2v, gvl );
		__builtin_epi_vstore_1xf64( (y0 + 3*n_elem_per_reg), y3v, gvl );

		x0 += n_elem_per_reg * n_iter_unroll;
		y0 += n_elem_per_reg * n_iter_unroll;
	}

	const double alphac = *alpha;

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const double x0c = *x0;

		*y0 += alphac * x0c;

		x0 += incx;
		y0 += incy;
	}

}

