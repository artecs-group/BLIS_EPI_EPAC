/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2018 - 2019, Advanced Micro Devices, Inc.
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

// -----------------------------------------------------------------------------

void bli_samaxv_epi_int
     (
       dim_t            n,
       float*  restrict x, inc_t incx,
       dim_t*  restrict i_max,
       cntx_t* restrict cntx
     )
{
	float*  minus_one = PASTEMAC(s,m1);
	dim_t*  zero_i    = PASTEMAC(i,0);

	float   chi1_r;
	float   abs_chi1;
	float   abs_chi1_max;
	uint32_t   i_max_l;
	uint32_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *i_max );
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(s,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 8 )
	{
		for ( i = 0; i < n; ++i )
		{
			float* chi1 = x + (i  )*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabsf( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */
			if ( abs_chi1_max < abs_chi1 || isnan( abs_chi1 ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		uint32_t  n_iter, n_left;
		uint32_t  num_vec_elements = 8;
		__epi_2xf32 x_vec, max_vec;
		__epi_2xi32 maxInx_vec;
		__epi_2xi1 mask_vec;
		__epi_2xi32 idx_vec, inc_vec;

		__epi_2xf32 max_vec_lo, max_vec_hi;
		__epi_2xi1 mask_vec_lo;
		__epi_2xi32 maxInx_vec_lo, maxInx_vec_hi;
		__epi_2xi32 permute_idx_vec;

		__epi_2xf32 vv;	
		__epi_2xi32 uu;

		long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );

		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		int idxs[8]       = {0, 1, 2, 3, 4, 5, 6, 7 };
		idx_vec	            = __builtin_epi_vload_2xi32( idxs, gvl);

		int permute_idxs[4] = {2, 3, 0, 1 };
		permute_idx_vec	    = __builtin_epi_vload_2xi32( permute_idxs, gvl);

		inc_vec             = __builtin_epi_vmv_v_x_2xi32(8, gvl); 
		max_vec             = __builtin_epi_vfmv_v_f_2xf32(-1.0, gvl); 
		maxInx_vec          = __builtin_epi_vmv_v_x_2xi32(0, gvl); 

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec      = __builtin_epi_vload_2xf32( x, gvl );


			// Get the absolute value of the vector element.
			x_vec      = __builtin_epi_vfsgnjx_2xf32( x_vec, x_vec, gvl );

			mask_vec  = __builtin_epi_vmfgt_2xf32( x_vec, max_vec, gvl );

			max_vec    = __builtin_epi_vfmerge_2xf32( max_vec, x_vec, mask_vec, gvl );
			
			maxInx_vec = __builtin_epi_vmerge_2xi32( maxInx_vec, idx_vec, mask_vec, gvl );

			idx_vec = __builtin_epi_vadd_2xi32(idx_vec, inc_vec, gvl);
			x       += num_vec_elements;
		}

		max_vec_lo  = __builtin_epi_vslideup_2xf32(max_vec, gvl / 2, gvl);
		max_vec_lo  = __builtin_epi_vslidedown_2xf32(max_vec_lo, gvl / 2, gvl);
		max_vec_hi  = __builtin_epi_vslidedown_2xf32(max_vec, gvl / 2, gvl);
		mask_vec_lo   = __builtin_epi_vmfgt_2xf32( max_vec_hi, max_vec_lo, gvl );

		max_vec_lo    = __builtin_epi_vfmerge_2xf32( max_vec_lo, max_vec_hi, mask_vec_lo, gvl );

		maxInx_vec_lo  = __builtin_epi_vslideup_2xi32(maxInx_vec, gvl / 2, gvl);;
		maxInx_vec_lo  = __builtin_epi_vslidedown_2xi32(maxInx_vec_lo, gvl / 2, gvl);;
		maxInx_vec_hi  = __builtin_epi_vslidedown_2xi32( maxInx_vec, gvl / 2,  gvl );
		maxInx_vec_lo   = __builtin_epi_vmerge_2xi32( maxInx_vec_lo, maxInx_vec_hi, mask_vec_lo, gvl );

		max_vec_hi    = __builtin_epi_vrgather_2xf32( max_vec_lo, permute_idx_vec, gvl );
		maxInx_vec_hi = __builtin_epi_vrgather_2xi32( maxInx_vec_lo, permute_idx_vec, gvl );

		mask_vec_lo   = __builtin_epi_vmfgt_2xf32( max_vec_hi, max_vec_lo, gvl );

		max_vec_lo    = __builtin_epi_vfmerge_2xf32( max_vec_lo, max_vec_hi, mask_vec_lo, gvl );
		maxInx_vec_lo   = __builtin_epi_vmerge_2xi32( maxInx_vec_lo, maxInx_vec_hi, mask_vec_lo, gvl );
 
		vv = __builtin_epi_vslidedown_2xf32(max_vec_lo, 0, gvl);
		float max_vec_lo_f0    = __builtin_epi_vfmv_f_s_2xf32(vv); 
	
		vv = __builtin_epi_vslidedown_2xf32(max_vec_lo, 1, gvl);
		float max_vec_lo_f1    = __builtin_epi_vfmv_f_s_2xf32(vv); 
		
		uu = __builtin_epi_vslidedown_2xi32(maxInx_vec_lo, 0, gvl);
		int   maxInx_vec_lo_f0 = __builtin_epi_vmv_x_s_2xi32(uu); 
		
		uu = __builtin_epi_vslidedown_2xi32(maxInx_vec_lo, 1, gvl);
		int   maxInx_vec_lo_f1 = __builtin_epi_vmv_x_s_2xi32(uu);
	
		if ( max_vec_lo_f0 > max_vec_lo_f1 )
		{
			abs_chi1_max = max_vec_lo_f0;
			i_max_l      = maxInx_vec_lo_f0;
		}
		else
		{
			abs_chi1_max = max_vec_lo_f1;
			i_max_l      = maxInx_vec_lo_f1;
		}

		for ( i = n - n_left; i < n; i++ )
		{
			float* chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabsf( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */
			if ( abs_chi1_max < abs_chi1 || isnan( abs_chi1 ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}


	/* Store final index to output variable. */
	*i_max = i_max_l;
}

// -----------------------------------------------------------------------------


void bli_damaxv_epi_int
     (
       dim_t            n,
       double* restrict x, inc_t incx,
       dim_t*  restrict i_max,
       cntx_t* restrict cntx
     )
{
	double* minus_one = PASTEMAC(d,m1);
	dim_t*  zero_i    = PASTEMAC(i,0);

	double  chi1_r;
	double  abs_chi1;
	double  abs_chi1_max;
	uint32_t   i_max_l;
	uint32_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *i_max );
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(d,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 4 )
	{
		for ( i = 0; i < n; ++i )
		{
			double* chi1 = x + (i  )*incx;

		/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabs( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */
			if ( abs_chi1_max < abs_chi1 || isnan( abs_chi1 ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		uint32_t  n_iter, n_left;
		uint32_t  num_vec_elements = 4;
		__epi_1xf64 x_vec, max_vec;
	        __epi_1xi64 maxInx_vec;
		__epi_1xi1  mask_vec;
		__epi_1xi64 idx_vec, inc_vec;

		__epi_1xf64 max_vec_lo, max_vec_hi;
		__epi_1xi1 mask_vec_lo;
		__epi_1xi64 maxInx_vec_lo, maxInx_vec_hi;

		__epi_1xf64 vv;
		__epi_1xi64 uu;

		long gvl = __builtin_epi_vsetvl( 4, __epi_e64, __epi_m1 );
		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		long int idxs[4]       = {0, 1, 2, 3 };
		idx_vec	            = __builtin_epi_vload_1xi64( idxs, gvl);
		inc_vec    = __builtin_epi_vmv_v_x_1xi64( 4, gvl );
		max_vec    = __builtin_epi_vfmv_v_f_1xf64( -1, gvl );
		maxInx_vec = __builtin_epi_vmv_v_x_1xi64( 0, gvl);

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec      = __builtin_epi_vload_1xf64( x, gvl );

			// Get the absolute value of the vector element.
			x_vec      = __builtin_epi_vfsgnjx_1xf64( x_vec, x_vec, gvl );

			mask_vec   = __builtin_epi_vmfgt_1xf64( x_vec, max_vec, gvl );

			max_vec    = __builtin_epi_vfmerge_1xf64( max_vec, x_vec, mask_vec, gvl);
			maxInx_vec = __builtin_epi_vmerge_1xi64( maxInx_vec, idx_vec, mask_vec, gvl );

			idx_vec =  __builtin_epi_vadd_1xi64(idx_vec, inc_vec, gvl);
			x         += num_vec_elements;
		}

		max_vec_lo  = __builtin_epi_vslideup_1xf64(max_vec, gvl / 2, gvl);
		max_vec_lo  = __builtin_epi_vslidedown_1xf64(max_vec_lo, gvl / 2, gvl);
		max_vec_hi  = __builtin_epi_vslidedown_1xf64(max_vec, gvl / 2, gvl);
		mask_vec_lo   = __builtin_epi_vmfgt_1xf64( max_vec_hi, max_vec_lo, gvl );

		max_vec_lo    = __builtin_epi_vfmerge_1xf64( max_vec_lo, max_vec_hi, mask_vec_lo, gvl );

		maxInx_vec_lo  = __builtin_epi_vslideup_1xi64(maxInx_vec, gvl / 2, gvl);;
		maxInx_vec_lo  = __builtin_epi_vslidedown_1xi64(maxInx_vec_lo, gvl / 2, gvl);;
		maxInx_vec_hi  = __builtin_epi_vslidedown_1xi64( maxInx_vec, gvl / 2,  gvl );
		maxInx_vec_lo   = __builtin_epi_vmerge_1xi64( maxInx_vec_lo, maxInx_vec_hi, mask_vec_lo, gvl );

		vv = __builtin_epi_vslidedown_1xf64(max_vec_lo, 0, gvl);
		double max_vec_lo_d0    = __builtin_epi_vfmv_f_s_1xf64(vv);
	
       		vv = __builtin_epi_vslidedown_1xf64(max_vec_lo, 1, gvl);
		double max_vec_lo_d1    = __builtin_epi_vfmv_f_s_1xf64(vv); 

		uu = __builtin_epi_vslidedown_1xi64(maxInx_vec_lo, 0, gvl);
		int   maxInx_vec_lo_d0 = __builtin_epi_vmv_x_s_1xi64(uu); 
	
       		uu = __builtin_epi_vslidedown_1xi64(maxInx_vec_lo, 1, gvl);
		int   maxInx_vec_lo_d1 = __builtin_epi_vmv_x_s_1xi64(uu); 
		
		if ( max_vec_lo_d0 > max_vec_lo_d1 )
		{
			abs_chi1_max = max_vec_lo_d0;
			i_max_l      = maxInx_vec_lo_d0;
		}
		else
		{
			abs_chi1_max = max_vec_lo_d1;
			i_max_l      = maxInx_vec_lo_d1;
		}

		for ( i = n - n_left; i < n; i++ )
		{
			double* chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabs( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */
			if ( abs_chi1_max < abs_chi1 || isnan( abs_chi1 ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}


	/* Store final index to output variable. */
	*i_max = i_max_l;
}

