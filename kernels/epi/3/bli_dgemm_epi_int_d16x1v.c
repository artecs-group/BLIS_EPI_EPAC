#include "blis.h"

//#include "vehave-control.h"

// -I development/include/vehave
// vehave_trace(1000, 1);
// vehave_trace(1000, 0);

void bli_dgemm_epi_scalar_16x1v
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       double*     restrict alpha,
       double*     restrict a,
       double*     restrict b,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint32_t k_iter = k / 1;
	//uint32_t k_left = k % 1;
	uint32_t rs_c   = rs_c0;
	uint32_t cs_c   = cs_c0;
	uint32_t i;

	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

	//const dim_t     mr     = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ); 
        const dim_t     nr     = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ); 

	GEMM_UKR_SETUP_CT_AMBI( d, bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ),
        		      bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ),
			      true ); 

	//printf("GEMM MICROKERNEL. M: %d, N: %d, K: %d - cs_c: %d, rs_c: %d - alpha: %f, beta: %f\n", m, n, k, cs_c, rs_c, *alpha, *beta);

	//long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );
	//long gvl = __builtin_epi_vsetvl( mr, __epi_e64, __epi_m1 );
	long gvl = __builtin_epi_vsetvl( nr, __epi_e64, __epi_m1 );

        unsigned long int vlen = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);

	__epi_1xf64 alphav;
	alphav = __builtin_epi_vfmv_v_f_1xf64( *alpha, gvl );

	// B vectors.
	__epi_1xf64 bv00; //, av01, av02, av03;

	// C rows (16).
	//             0,    1,    2,    3,    4,    5,    6,    7
	__epi_1xf64 cv00, 
		    cv10, 
		    cv20, 
		    cv30, 
		    cv40, 
		    cv50, 
		    cv60, 
		    cv70;
	//             8,    9,    10,    11,    12,    13,    14,    15
	__epi_1xf64 cv80, 
		    cv90, 
		    cva0, 
		    cvb0, 
		    cvc0, 
		    cvd0, 
		    cve0, 
		    cvf0;

	// Accummulators (16).
	//              0,     1,     2,     3,     4,     5,     6,     7
	__epi_1xf64 abv00, 
		    abv10, 
		    abv20, 
		    abv30, 
		    abv40, 
		    abv50, 
		    abv60, 
		    abv70;
	//             8,      9,     10,     11,     12,     13,     14,     15
	__epi_1xf64 abv80, 
		    abv90, 
		    abva0, 
		    abvb0, 
		    abvc0, 
		    abvd0, 
		    abve0, 
		    abvf0;


	// Initialize accummulators to 0.0 (column 0)
	abv00 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 1)
	abv10 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 2)
	abv20 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 3)
	abv30 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 4)
	abv40 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 5)
	abv50 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 6)
	abv60 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 7)
	abv70 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 8)
	abv80 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 9)
	abv90 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 10)
	abva0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 11)
	abvb0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 12)
	abvc0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 13)
	abvd0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 14)
	abve0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 15)
	abvf0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );

	__epi_1xf64 sav1;

	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iteration 0
 		bv00 = __builtin_epi_vload_1xf64( b+0*vlen, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+1), gvl );
		abv10 = __builtin_epi_vfmacc_1xf64( abv10, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+2), gvl );
		abv20 = __builtin_epi_vfmacc_1xf64( abv20, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+3), gvl );
		abv30 = __builtin_epi_vfmacc_1xf64( abv30, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+4), gvl );
		abv40 = __builtin_epi_vfmacc_1xf64( abv40, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+5), gvl );
		abv50 = __builtin_epi_vfmacc_1xf64( abv50, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+6), gvl );
		abv60 = __builtin_epi_vfmacc_1xf64( abv60, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+7), gvl );
		abv70 = __builtin_epi_vfmacc_1xf64( abv70, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+8), gvl );
		abv80 = __builtin_epi_vfmacc_1xf64( abv80, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+9), gvl );
		abv90 = __builtin_epi_vfmacc_1xf64( abv90, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+10), gvl );
		abva0 = __builtin_epi_vfmacc_1xf64( abva0, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+11), gvl );
		abvb0 = __builtin_epi_vfmacc_1xf64( abvb0, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+12), gvl );
		abvc0 = __builtin_epi_vfmacc_1xf64( abvc0, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+13), gvl );
		abvd0 = __builtin_epi_vfmacc_1xf64( abvd0, sav1, bv00, gvl );

 		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+14), gvl );
		abve0 = __builtin_epi_vfmacc_1xf64( abve0, sav1, bv00, gvl );

		sav1 = __builtin_epi_vfmv_v_f_1xf64( *(a+15), gvl );
		abvf0 = __builtin_epi_vfmacc_1xf64( abvf0, sav1, bv00, gvl );

	        // Adjust pointers for next iterations.
	        b += 1 * vlen * 1; // 1 vectors + 1 Unroll factor
		a += 16;
	}

#if 0
	for ( i = 0; i < k_left; ++i )
	{
 		av00 = __builtin_epi_vload_1xf64( a+0*vlen, gvl );

 		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sbv1, av00, gvl );

		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+1), gvl );
		abv01 = __builtin_epi_vfmacc_1xf64( abv01, sbv1, av00, gvl );

 		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+2), gvl );
		abv02 = __builtin_epi_vfmacc_1xf64( abv02, sbv1, av00, gvl );

		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+3), gvl );
		abv03 = __builtin_epi_vfmacc_1xf64( abv03, sbv1, av00, gvl );

 		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+4), gvl );
		abv04 = __builtin_epi_vfmacc_1xf64( abv04, sbv1, av00, gvl );

		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+5), gvl );
		abv05 = __builtin_epi_vfmacc_1xf64( abv05, sbv1, av00, gvl );

 		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+6), gvl );
		abv06 = __builtin_epi_vfmacc_1xf64( abv06, sbv1, av00, gvl );

		sbv1 = __builtin_epi_vfmv_v_f_1xf64( *(b+7), gvl );
		abv07 = __builtin_epi_vfmacc_1xf64( abv07, sbv1, av00, gvl );

		a += 1*vlen;
		b += 24;
	}
#endif

	if ( *beta != 0.0 ) {

	// Row-major.
	if( cs_c == 1 )
	{
		//printf("cs_c == 1\n");
		// Load row 0
 		cv00 = __builtin_epi_vload_1xf64( c + 0*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 1
	      	cv10 = __builtin_epi_vload_1xf64( c + 1*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 2
	       	cv20 = __builtin_epi_vload_1xf64( c + 2*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 3
	       	cv30 = __builtin_epi_vload_1xf64( c + 3*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 4
	       	cv40 = __builtin_epi_vload_1xf64( c + 4*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 5
	     	cv50 = __builtin_epi_vload_1xf64( c + 5*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 6
	        cv60 = __builtin_epi_vload_1xf64( c + 6*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 7
	      	cv70 = __builtin_epi_vload_1xf64( c + 7*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 8
 		cv80 = __builtin_epi_vload_1xf64( c + 8*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 9
	      	cv90 = __builtin_epi_vload_1xf64( c + 9*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 10
	       	cva0 = __builtin_epi_vload_1xf64( c + 10*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 11
	       	cvb0 = __builtin_epi_vload_1xf64( c + 11*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 12
	       	cvc0 = __builtin_epi_vload_1xf64( c + 12*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 13
	     	cvd0 = __builtin_epi_vload_1xf64( c + 13*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 14
	        cve0 = __builtin_epi_vload_1xf64( c + 14*rs_c + 0*cs_c + 0*vlen, gvl );

		// Load row 15
	      	cvf0 = __builtin_epi_vload_1xf64( c + 15*rs_c + 0*cs_c + 0*vlen, gvl );
	}
	// Column major.
	if( rs_c == 1 )
	{
		//printf("rs_c == 1\n");
		// Load row 0
 		cv00 = __builtin_epi_vload_strided_1xf64( c + 0*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 1
	      	cv10 = __builtin_epi_vload_strided_1xf64( c + 1*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 2
	       	cv20 = __builtin_epi_vload_strided_1xf64( c + 2*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 3
	       	cv30 = __builtin_epi_vload_strided_1xf64( c + 3*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 4
	       	cv40 = __builtin_epi_vload_strided_1xf64( c + 4*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 5
	     	cv50 = __builtin_epi_vload_strided_1xf64( c + 5*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 6
	        cv60 = __builtin_epi_vload_strided_1xf64( c + 6*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 7
	      	cv70 = __builtin_epi_vload_strided_1xf64( c + 7*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 8
 		cv80 = __builtin_epi_vload_strided_1xf64( c + 8*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 9
	      	cv90 = __builtin_epi_vload_strided_1xf64( c + 9*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 10
	       	cva0 = __builtin_epi_vload_strided_1xf64( c + 10*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 11
	       	cvb0 = __builtin_epi_vload_strided_1xf64( c + 11*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 12
	       	cvc0 = __builtin_epi_vload_strided_1xf64( c + 12*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 13
	     	cvd0 = __builtin_epi_vload_strided_1xf64( c + 13*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 14
	        cve0 = __builtin_epi_vload_strided_1xf64( c + 14*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );

		// Load row 15
	      	cvf0 = __builtin_epi_vload_strided_1xf64( c + 15*rs_c + 0*cs_c + 0*vlen, (long int)cs_c*sizeof(double) , gvl );
	}

	__epi_1xf64 betav;
	betav = __builtin_epi_vfmv_v_f_1xf64( *beta, gvl );

	cv00 = __builtin_epi_vfmul_1xf64( cv00, betav, gvl );
	cv10 = __builtin_epi_vfmul_1xf64( cv10, betav, gvl );
	cv20 = __builtin_epi_vfmul_1xf64( cv20, betav, gvl );
	cv30 = __builtin_epi_vfmul_1xf64( cv30, betav, gvl );
	cv40 = __builtin_epi_vfmul_1xf64( cv40, betav, gvl );
	cv50 = __builtin_epi_vfmul_1xf64( cv50, betav, gvl );
	cv60 = __builtin_epi_vfmul_1xf64( cv60, betav, gvl );
	cv70 = __builtin_epi_vfmul_1xf64( cv70, betav, gvl );
	cv80 = __builtin_epi_vfmul_1xf64( cv80, betav, gvl );
	cv90 = __builtin_epi_vfmul_1xf64( cv90, betav, gvl );
	cva0 = __builtin_epi_vfmul_1xf64( cva0, betav, gvl );
	cvb0 = __builtin_epi_vfmul_1xf64( cvb0, betav, gvl );
	cvc0 = __builtin_epi_vfmul_1xf64( cvc0, betav, gvl );
	cvd0 = __builtin_epi_vfmul_1xf64( cvd0, betav, gvl );
	cve0 = __builtin_epi_vfmul_1xf64( cve0, betav, gvl );
	cvf0 = __builtin_epi_vfmul_1xf64( cvf0, betav, gvl );

	}
	else { // Beta == 0
	  cv00 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv10 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv20 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv30 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv40 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv50 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv60 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv70 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv80 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cv90 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cva0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cvb0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cvc0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cvd0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cve0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	  cvf0 = __builtin_epi_vfmv_v_f_1xf64( 0.0, gvl );
	}

	//This segfaults if cv0 = .... ( cv0, ...
	cv00 = __builtin_epi_vfmacc_1xf64( cv00, alphav, abv00, gvl );
	cv10 = __builtin_epi_vfmacc_1xf64( cv10, alphav, abv10, gvl );
	cv20 = __builtin_epi_vfmacc_1xf64( cv20, alphav, abv20, gvl );
	cv30 = __builtin_epi_vfmacc_1xf64( cv30, alphav, abv30, gvl );
	cv40 = __builtin_epi_vfmacc_1xf64( cv40, alphav, abv40, gvl );
	cv50 = __builtin_epi_vfmacc_1xf64( cv50, alphav, abv50, gvl );
	cv60 = __builtin_epi_vfmacc_1xf64( cv60, alphav, abv60, gvl );
	cv70 = __builtin_epi_vfmacc_1xf64( cv70, alphav, abv70, gvl );
	cv80 = __builtin_epi_vfmacc_1xf64( cv80, alphav, abv80, gvl );
	cv90 = __builtin_epi_vfmacc_1xf64( cv90, alphav, abv90, gvl );
	cva0 = __builtin_epi_vfmacc_1xf64( cva0, alphav, abva0, gvl );
	cvb0 = __builtin_epi_vfmacc_1xf64( cvb0, alphav, abvb0, gvl );
	cvc0 = __builtin_epi_vfmacc_1xf64( cvc0, alphav, abvc0, gvl );
	cvd0 = __builtin_epi_vfmacc_1xf64( cvd0, alphav, abvd0, gvl );
	cve0 = __builtin_epi_vfmacc_1xf64( cve0, alphav, abve0, gvl );
	cvf0 = __builtin_epi_vfmacc_1xf64( cvf0, alphav, abvf0, gvl );

	// Row-major.
	if( cs_c == 1 )
	{
		//printf("cs_c == 1\n");
		// Store row 0
		__builtin_epi_vstore_1xf64( c + 0*rs_c + 0*cs_c+0*vlen,    cv00, gvl );
		// Store row 1
		__builtin_epi_vstore_1xf64( c + 1*rs_c + 0*cs_c+0*vlen,    cv10, gvl );
		// Store row 2
		__builtin_epi_vstore_1xf64( c + 2*rs_c + 0*cs_c+0*vlen,    cv20, gvl );
		// Store row 3
		__builtin_epi_vstore_1xf64( c + 3*rs_c + 0*cs_c+0*vlen,    cv30, gvl );
		// Store row 4
		__builtin_epi_vstore_1xf64( c + 4*rs_c + 0*cs_c+0*vlen,    cv40, gvl );
		// Store row 5
		__builtin_epi_vstore_1xf64( c + 5*rs_c + 0*cs_c+0*vlen,    cv50, gvl );
		// Store row 6
		__builtin_epi_vstore_1xf64( c + 6*rs_c + 0*cs_c+0*vlen,    cv60, gvl );
		// Store row 7
		__builtin_epi_vstore_1xf64( c + 7*rs_c + 0*cs_c+0*vlen,    cv70, gvl );
                // Store row 8
		__builtin_epi_vstore_1xf64( c + 8*rs_c + 0*cs_c+0*vlen,    cv80, gvl );
		// Store row 9
		__builtin_epi_vstore_1xf64( c + 9*rs_c + 0*cs_c+0*vlen,    cv90, gvl );
		// Store row 10
		__builtin_epi_vstore_1xf64( c + 10*rs_c + 0*cs_c+0*vlen,    cva0, gvl );
		// Store row 11
		__builtin_epi_vstore_1xf64( c + 11*rs_c + 0*cs_c+0*vlen,    cvb0, gvl );
		// Store row 12
		__builtin_epi_vstore_1xf64( c + 12*rs_c + 0*cs_c+0*vlen,    cvc0, gvl );
		// Store row 13
		__builtin_epi_vstore_1xf64( c + 13*rs_c + 0*cs_c+0*vlen,    cvd0, gvl );
		// Store row 14
		__builtin_epi_vstore_1xf64( c + 14*rs_c + 0*cs_c+0*vlen,    cve0, gvl );
		// Store row 15
		__builtin_epi_vstore_1xf64( c + 15*rs_c + 0*cs_c+0*vlen,    cvf0, gvl );

	}
	// Column-major.
	if( rs_c == 1 )
	{
		//printf("rs_c == 1\n");
		// Store row 0
		__builtin_epi_vstore_strided_1xf64( c + 0*rs_c + 0*cs_c+0*vlen,    cv00, (long int)cs_c*sizeof(double) , gvl );
		// Store row 1
		__builtin_epi_vstore_strided_1xf64( c + 1*rs_c + 0*cs_c+0*vlen,    cv10, (long int)cs_c*sizeof(double) , gvl );
		// Store row 2
		__builtin_epi_vstore_strided_1xf64( c + 2*rs_c + 0*cs_c+0*vlen,    cv20, (long int)cs_c*sizeof(double) , gvl );
		// Store row 3
		__builtin_epi_vstore_strided_1xf64( c + 3*rs_c + 0*cs_c+0*vlen,    cv30, (long int)cs_c*sizeof(double) , gvl );
		// Store row 4
		__builtin_epi_vstore_strided_1xf64( c + 4*rs_c + 0*cs_c+0*vlen,    cv40, (long int)cs_c*sizeof(double) , gvl );
		// Store row 5
		__builtin_epi_vstore_strided_1xf64( c + 5*rs_c + 0*cs_c+0*vlen,    cv50, (long int)cs_c*sizeof(double) , gvl );
		// Store row 6
		__builtin_epi_vstore_strided_1xf64( c + 6*rs_c + 0*cs_c+0*vlen,    cv60, (long int)cs_c*sizeof(double) , gvl );
		// Store row 7
		__builtin_epi_vstore_strided_1xf64( c + 7*rs_c + 0*cs_c+0*vlen,    cv70, (long int)cs_c*sizeof(double) , gvl );
                // Store row 8
		__builtin_epi_vstore_strided_1xf64( c + 8*rs_c + 0*cs_c+0*vlen,    cv80, (long int)cs_c*sizeof(double) , gvl );
		// Store row 9
		__builtin_epi_vstore_strided_1xf64( c + 9*rs_c + 0*cs_c+0*vlen,    cv90, (long int)cs_c*sizeof(double) , gvl );
		// Store row 10
		__builtin_epi_vstore_strided_1xf64( c + 10*rs_c + 0*cs_c+0*vlen,    cva0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 11
		__builtin_epi_vstore_strided_1xf64( c + 11*rs_c + 0*cs_c+0*vlen,    cvb0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 12
		__builtin_epi_vstore_strided_1xf64( c + 12*rs_c + 0*cs_c+0*vlen,    cvc0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 13
		__builtin_epi_vstore_strided_1xf64( c + 13*rs_c + 0*cs_c+0*vlen,    cvd0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 14
		__builtin_epi_vstore_strided_1xf64( c + 14*rs_c + 0*cs_c+0*vlen,    cve0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 15
		__builtin_epi_vstore_strided_1xf64( c + 15*rs_c + 0*cs_c+0*vlen,    cvf0, (long int)cs_c*sizeof(double) , gvl );

	}

	GEMM_UKR_FLUSH_CT( d );

}

