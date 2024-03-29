#include "blis.h"

#ifdef EPI_FPGA
#define BROADCAST_f64 __builtin_epi_vbroadcast_1xf64
#else
#define BROADCAST_f64 __builtin_epi_vfmv_v_f_1xf64
#endif

void bli_dgemmtrsm_l_epi_scalar_16x1v
     (
       dim_t                m,
       dim_t                n,
       dim_t                k0,
       double*     restrict alpha,
       double*     restrict a10,
       double*     restrict a11,
       double*     restrict b01,
       double*     restrict b11,
       double*     restrict c11, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint32_t k_iter = k0 / 1;
	//uint32_t k_left = k0 % 1;
	uint32_t rs_c   = rs_c0;
	uint32_t cs_c   = cs_c0;
	uint32_t i;

	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

	const dim_t     mr     = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ); 
        const dim_t     nr     = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ); 

	GEMMTRSM_UKR_SETUP_CT_AMBI( d, bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ),
                                  bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ),
				  true );

	long gvl = __builtin_epi_vsetvl( nr, __epi_e64, __epi_m1 );

        unsigned long int vlen = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);

	// Load alpha and duplicate.
	__epi_1xf64 alphav;
	alphav = BROADCAST_f64( *alpha, gvl );

	// B vectors.
	__epi_1xf64 bv00;

	// C rows (16).
	// 0,    1,    2,    3,    4,    5,    6,    7
	__epi_1xf64 cv00, 
		    cv10, 
		    cv20, 
		    cv30, 
		    cv40, 
		    cv50, 
		    cv60, 
		    cv70;
	// 8,    9,    10,    11,    12,    13,    14,    15
	__epi_1xf64 cv80, 
		    cv90, 
		    cva0, 
		    cvb0, 
		    cvc0, 
		    cvd0, 
		    cve0, 
		    cvf0;

	// Accummulators (16).
	// 0,     1,     2,     3,     4,     5,     6,     7
	__epi_1xf64 abv00, 
		    abv10, 
		    abv20, 
		    abv30, 
		    abv40, 
		    abv50, 
		    abv60, 
		    abv70;
	// 8,      9,     10,     11,     12,     13,     14,     15
	__epi_1xf64 abv80, 
		    abv90, 
		    abva0, 
		    abvb0, 
		    abvc0, 
		    abvd0, 
		    abve0, 
		    abvf0;


	// Initialize accummulators to 0.0 (row 0)
	abv00 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 1)
	abv10 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 2)
	abv20 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 3)
	abv30 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 4)
	abv40 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 5)
	abv50 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 6)
	abv60 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 7)
	abv70 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 8)
	abv80 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 9)
	abv90 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 10)
	abva0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 11)
	abvb0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 12)
	abvc0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 13)
	abvd0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 14)
	abve0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 15)
	abvf0 = BROADCAST_f64( 0.0, gvl );

	__epi_1xf64 sav1;

	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iteration 0
 		bv00 = __builtin_epi_vload_1xf64( b01+0*vlen, gvl );

 		sav1 = BROADCAST_f64( *(a10), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+1), gvl );
		abv10 = __builtin_epi_vfmacc_1xf64( abv10, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+2), gvl );
		abv20 = __builtin_epi_vfmacc_1xf64( abv20, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+3), gvl );
		abv30 = __builtin_epi_vfmacc_1xf64( abv30, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+4), gvl );
		abv40 = __builtin_epi_vfmacc_1xf64( abv40, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+5), gvl );
		abv50 = __builtin_epi_vfmacc_1xf64( abv50, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+6), gvl );
		abv60 = __builtin_epi_vfmacc_1xf64( abv60, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+7), gvl );
		abv70 = __builtin_epi_vfmacc_1xf64( abv70, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+8), gvl );
		abv80 = __builtin_epi_vfmacc_1xf64( abv80, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+9), gvl );
		abv90 = __builtin_epi_vfmacc_1xf64( abv90, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+10), gvl );
		abva0 = __builtin_epi_vfmacc_1xf64( abva0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+11), gvl );
		abvb0 = __builtin_epi_vfmacc_1xf64( abvb0, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+12), gvl );
		abvc0 = __builtin_epi_vfmacc_1xf64( abvc0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+13), gvl );
		abvd0 = __builtin_epi_vfmacc_1xf64( abvd0, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a10+14), gvl );
		abve0 = __builtin_epi_vfmacc_1xf64( abve0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a10+15), gvl );
		abvf0 = __builtin_epi_vfmacc_1xf64( abvf0, sav1, bv00, gvl );

	        // Adjust pointers for next iterations.
	        b01 += nr; // 1 vectors + 1 Unroll factor
		a10 += mr;
	}
	
#if 0
	for ( i = 0; i < k_left; ++i )
	{
 		av00 = __builtin_epi_vload_1xf64( a+0*vlen, gvl );

 		sbv1 = BROADCAST_f64( *(b), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sbv1, av00, gvl );

		sbv1 = BROADCAST_f64( *(b+1), gvl );
		abv01 = __builtin_epi_vfmacc_1xf64( abv01, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f64( *(b+2), gvl );
		abv02 = __builtin_epi_vfmacc_1xf64( abv02, sbv1, av00, gvl );

		sbv1 = BROADCAST_f64( *(b+3), gvl );
		abv03 = __builtin_epi_vfmacc_1xf64( abv03, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f64( *(b+4), gvl );
		abv04 = __builtin_epi_vfmacc_1xf64( abv04, sbv1, av00, gvl );

		sbv1 = BROADCAST_f64( *(b+5), gvl );
		abv05 = __builtin_epi_vfmacc_1xf64( abv05, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f64( *(b+6), gvl );
		abv06 = __builtin_epi_vfmacc_1xf64( abv06, sbv1, av00, gvl );

		sbv1 = BROADCAST_f64( *(b+7), gvl );
		abv07 = __builtin_epi_vfmacc_1xf64( abv07, sbv1, av00, gvl );

		a += 1*vlen;
		b += 24;
	}
#endif

	// b11 := alpha * b11 - a10 * b01
	//
	// Load alpha and duplicate

	// b11 := alpha * b11 - a10 * b01
 	cv00 = __builtin_epi_vload_1xf64( b11+0*nr, gvl );
	cv00 = __builtin_epi_vfmsub_1xf64( cv00, alphav, abv00, gvl );

 	cv10 = __builtin_epi_vload_1xf64( b11+1*nr, gvl );
	cv10 = __builtin_epi_vfmsub_1xf64( cv10, alphav, abv10, gvl );
	
 	cv20 = __builtin_epi_vload_1xf64( b11+2*nr, gvl );
	cv20 = __builtin_epi_vfmsub_1xf64( cv20, alphav, abv20, gvl );

 	cv30 = __builtin_epi_vload_1xf64( b11+3*nr, gvl );
	cv30 = __builtin_epi_vfmsub_1xf64( cv30, alphav, abv30, gvl );

 	cv40 = __builtin_epi_vload_1xf64( b11+4*nr, gvl );
	cv40 = __builtin_epi_vfmsub_1xf64( cv40, alphav, abv40, gvl );

 	cv50 = __builtin_epi_vload_1xf64( b11+5*nr, gvl );
	cv50 = __builtin_epi_vfmsub_1xf64( cv50, alphav, abv50, gvl );

 	cv60 = __builtin_epi_vload_1xf64( b11+6*nr, gvl );
	cv60 = __builtin_epi_vfmsub_1xf64( cv60, alphav, abv60, gvl );

 	cv70 = __builtin_epi_vload_1xf64( b11+7*nr, gvl );
	cv70 = __builtin_epi_vfmsub_1xf64( cv70, alphav, abv70, gvl );

 	cv80 = __builtin_epi_vload_1xf64( b11+8*nr, gvl );
	cv80 = __builtin_epi_vfmsub_1xf64( cv80, alphav, abv80, gvl );

 	cv90 = __builtin_epi_vload_1xf64( b11+9*nr, gvl );
	cv90 = __builtin_epi_vfmsub_1xf64( cv90, alphav, abv90, gvl );

 	cva0 = __builtin_epi_vload_1xf64( b11+10*nr, gvl );
	cva0 = __builtin_epi_vfmsub_1xf64( cva0, alphav, abva0, gvl );

 	cvb0 = __builtin_epi_vload_1xf64( b11+11*nr, gvl );
	cvb0 = __builtin_epi_vfmsub_1xf64( cvb0, alphav, abvb0, gvl );

 	cvc0 = __builtin_epi_vload_1xf64( b11+12*nr, gvl );
	cvc0 = __builtin_epi_vfmsub_1xf64( cvc0, alphav, abvc0, gvl );

 	cvd0 = __builtin_epi_vload_1xf64( b11+13*nr, gvl );
	cvd0 = __builtin_epi_vfmsub_1xf64( cvd0, alphav, abvd0, gvl );

 	cve0 = __builtin_epi_vload_1xf64( b11+14*nr, gvl );
	cve0 = __builtin_epi_vfmsub_1xf64( cve0, alphav, abve0, gvl );

 	cvf0 = __builtin_epi_vload_1xf64( b11+15*nr, gvl );
	cvf0 = __builtin_epi_vfmsub_1xf64( cvf0, alphav, abvf0, gvl );

	// Contents of b11 are stored by rows (16 rows, from cv00 to cvf0).
	// (cv00)
	// (cv10)
	// (cv20)
	// (cv30)
	// (cv40)
	// (cv50)
	// (cv60)
	// (cv70)
	// (cv80)
	// (cv90)
	// (cva0)
	// (cvb0)
	// (cvc0)
	// (cvd0)
	// (cve0)
	// (cvf0)
	//
	// Iteration 0  --------------------------------------------
	//
 	__epi_1xf64 alpha00 = BROADCAST_f64( *(a11), gvl ); // (1/alpha00)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv00 = __builtin_epi_vfmul_1xf64( cv00, alpha00, gvl ); // cv00 *= (1/alpha00)
#else
	cv00 = __builtin_epi_vfdiv_1xf64( cv00, alpha00, gvl ); // cv00 /= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 0*nr + 0*cs_c+0*vlen, cv00, gvl ); // Store row 1 of b11


	//
	// Iteration 1  --------------------------------------------
	//
	__epi_1xf64 alpha10 = BROADCAST_f64( *(a11 + 1 + 0 * 16), gvl ); // (alpha10)
	__epi_1xf64 alpha11 = BROADCAST_f64( *(a11 + 1 + 1 * 16), gvl ); // (1/alpha11)

	__epi_1xf64 accum = __builtin_epi_vfmul_1xf64( cv00, alpha10, gvl );  // Scale row 1 of B11

	cv10  = __builtin_epi_vfsub_1xf64( cv10, accum, gvl );    // Update row 2 of B11

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv10 = __builtin_epi_vfmul_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
#else
	cv10 = __builtin_epi_vfdiv_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 1*nr + 0*cs_c+0*vlen, cv10, gvl ); // Store row 2 of b11


	//
	// Iteration 2  --------------------------------------------
	//
	__epi_1xf64 alpha20 = BROADCAST_f64( *(a11 + 2 + 0 * 16), gvl ); // (alpha20)
	__epi_1xf64 alpha21 = BROADCAST_f64( *(a11 + 2 + 1 * 16), gvl ); // (alpha21)
	__epi_1xf64 alpha22 = BROADCAST_f64( *(a11 + 2 + 2 * 16), gvl ); // (1/alpha22)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha20, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha21, cv10, gvl );    // 

	cv20 = __builtin_epi_vfsub_1xf64( cv20, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv20 = __builtin_epi_vfmul_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
#else
	cv20 = __builtin_epi_vfdiv_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 2*nr + 0*cs_c+0*vlen, cv20, gvl ); // Store row 3 of b11

	//
	// Iteration 3  --------------------------------------------
	//
	__epi_1xf64 alpha30 = BROADCAST_f64( *(a11 + 3 + 0 * 16), gvl ); // (alpha30)
	__epi_1xf64 alpha31 = BROADCAST_f64( *(a11 + 3 + 1 * 16), gvl ); // (alpha31)
	__epi_1xf64 alpha32 = BROADCAST_f64( *(a11 + 3 + 2 * 16), gvl ); // (alpha32)
	__epi_1xf64 alpha33 = BROADCAST_f64( *(a11 + 3 + 3 * 16), gvl ); // (1/alpha33)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha30, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha31, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha32, cv20, gvl );    // 

	cv30 = __builtin_epi_vfsub_1xf64( cv30, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv30 = __builtin_epi_vfmul_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
#else
	cv30 = __builtin_epi_vfdiv_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 3*nr + 0*cs_c+0*vlen, cv30, gvl ); // Store row 4 of b11


	//
	// Iteration 4  --------------------------------------------
	//
	__epi_1xf64 alpha40 = BROADCAST_f64( *(a11 + 4 + 0 * 16), gvl ); // (alpha40)
	__epi_1xf64 alpha41 = BROADCAST_f64( *(a11 + 4 + 1 * 16), gvl ); // (alpha41)
	__epi_1xf64 alpha42 = BROADCAST_f64( *(a11 + 4 + 2 * 16), gvl ); // (alpha42)
	__epi_1xf64 alpha43 = BROADCAST_f64( *(a11 + 4 + 3 * 16), gvl ); // (alpha43)
	__epi_1xf64 alpha44 = BROADCAST_f64( *(a11 + 4 + 4 * 16), gvl ); // (1/alpha44)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha40, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha41, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha42, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha43, cv30, gvl );    // 

	cv40 = __builtin_epi_vfsub_1xf64( cv40, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv40 = __builtin_epi_vfmul_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
#else
	cv40 = __builtin_epi_vfdiv_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 4*nr + 0*cs_c+0*vlen, cv40, gvl ); // Store row 5 of b11


	//
	// Iteration 5  --------------------------------------------
	//
	__epi_1xf64 alpha50 = BROADCAST_f64( *(a11 + 5 + 0 * 16), gvl ); // (alpha50)
	__epi_1xf64 alpha51 = BROADCAST_f64( *(a11 + 5 + 1 * 16), gvl ); // (alpha51)
	__epi_1xf64 alpha52 = BROADCAST_f64( *(a11 + 5 + 2 * 16), gvl ); // (alpha52)
	__epi_1xf64 alpha53 = BROADCAST_f64( *(a11 + 5 + 3 * 16), gvl ); // (alpha53)
	__epi_1xf64 alpha54 = BROADCAST_f64( *(a11 + 5 + 4 * 16), gvl ); // (alpha54)
	__epi_1xf64 alpha55 = BROADCAST_f64( *(a11 + 5 + 5 * 16), gvl ); // (1/alpha55)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha50, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha51, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha52, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha53, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha54, cv40, gvl );    // 

	cv50 = __builtin_epi_vfsub_1xf64( cv50, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv50 = __builtin_epi_vfmul_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
#else
	cv50 = __builtin_epi_vfdiv_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 5*nr + 0*cs_c+0*vlen, cv50, gvl ); // Store row 6 of b11


	//
	// Iteration 6  --------------------------------------------
	//
	__epi_1xf64 alpha60 = BROADCAST_f64( *(a11 + 6 + 0 * 16), gvl ); // (alpha60)
	__epi_1xf64 alpha61 = BROADCAST_f64( *(a11 + 6 + 1 * 16), gvl ); // (alpha61)
	__epi_1xf64 alpha62 = BROADCAST_f64( *(a11 + 6 + 2 * 16), gvl ); // (alpha62)
	__epi_1xf64 alpha63 = BROADCAST_f64( *(a11 + 6 + 3 * 16), gvl ); // (alpha63)
	__epi_1xf64 alpha64 = BROADCAST_f64( *(a11 + 6 + 4 * 16), gvl ); // (alpha64)
	__epi_1xf64 alpha65 = BROADCAST_f64( *(a11 + 6 + 5 * 16), gvl ); // (alpha65)
	__epi_1xf64 alpha66 = BROADCAST_f64( *(a11 + 6 + 6 * 16), gvl ); // (1/alpha66)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha60, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha61, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha62, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha63, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha64, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha65, cv50, gvl );    // 

	cv60 = __builtin_epi_vfsub_1xf64( cv60, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv60 = __builtin_epi_vfmul_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
#else
	cv60 = __builtin_epi_vfdiv_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 6*nr + 0*cs_c+0*vlen, cv60, gvl ); // Store row 7 of b11



	//
	// Iteration 7  --------------------------------------------
	//
	__epi_1xf64 alpha70 = BROADCAST_f64( *(a11 + 7 + 0 * 16), gvl ); // (alpha70)
	__epi_1xf64 alpha71 = BROADCAST_f64( *(a11 + 7 + 1 * 16), gvl ); // (alpha71)
	__epi_1xf64 alpha72 = BROADCAST_f64( *(a11 + 7 + 2 * 16), gvl ); // (alpha72)
	__epi_1xf64 alpha73 = BROADCAST_f64( *(a11 + 7 + 3 * 16), gvl ); // (alpha73)
	__epi_1xf64 alpha74 = BROADCAST_f64( *(a11 + 7 + 4 * 16), gvl ); // (alpha74)
	__epi_1xf64 alpha75 = BROADCAST_f64( *(a11 + 7 + 5 * 16), gvl ); // (alpha75)
	__epi_1xf64 alpha76 = BROADCAST_f64( *(a11 + 7 + 6 * 16), gvl ); // (alpha76)
	__epi_1xf64 alpha77 = BROADCAST_f64( *(a11 + 7 + 7 * 16), gvl ); // (1/alpha77)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha70, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha71, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha72, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha73, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha74, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha75, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha76, cv60, gvl );    // 

	cv70 = __builtin_epi_vfsub_1xf64( cv70, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv70 = __builtin_epi_vfmul_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
#else
	cv70 = __builtin_epi_vfdiv_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 7*nr + 0*cs_c+0*vlen, cv70, gvl ); // Store row 8 of b11


	//
	// Iteration 8  --------------------------------------------
	//
	__epi_1xf64 alpha80 = BROADCAST_f64( *(a11 + 8 + 0 * 16), gvl ); // (alpha80)
	__epi_1xf64 alpha81 = BROADCAST_f64( *(a11 + 8 + 1 * 16), gvl ); // (alpha81)
	__epi_1xf64 alpha82 = BROADCAST_f64( *(a11 + 8 + 2 * 16), gvl ); // (alpha82)
	__epi_1xf64 alpha83 = BROADCAST_f64( *(a11 + 8 + 3 * 16), gvl ); // (alpha83)
	__epi_1xf64 alpha84 = BROADCAST_f64( *(a11 + 8 + 4 * 16), gvl ); // (alpha84)
	__epi_1xf64 alpha85 = BROADCAST_f64( *(a11 + 8 + 5 * 16), gvl ); // (alpha85)
	__epi_1xf64 alpha86 = BROADCAST_f64( *(a11 + 8 + 6 * 16), gvl ); // (alpha86)
	__epi_1xf64 alpha87 = BROADCAST_f64( *(a11 + 8 + 7 * 16), gvl ); // (alpha87)
	__epi_1xf64 alpha88 = BROADCAST_f64( *(a11 + 8 + 8 * 16), gvl ); // (1/alpha88)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha80, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha81, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha82, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha83, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha84, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha85, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha86, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha87, cv70, gvl );    // 

	cv80 = __builtin_epi_vfsub_1xf64( cv80, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv80 = __builtin_epi_vfmul_1xf64( cv80, alpha88, gvl ); // abv0 *= alpha00
#else
	cv80 = __builtin_epi_vfdiv_1xf64( cv80, alpha88, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 8*nr + 0*cs_c+0*vlen, cv80, gvl ); // Store row 9 of b11


	//
	// Iteration 9  --------------------------------------------
	//
	__epi_1xf64 alpha90 = BROADCAST_f64( *(a11 + 9 + 0 * 16), gvl ); // (alpha90)
	__epi_1xf64 alpha91 = BROADCAST_f64( *(a11 + 9 + 1 * 16), gvl ); // (alpha91)
	__epi_1xf64 alpha92 = BROADCAST_f64( *(a11 + 9 + 2 * 16), gvl ); // (alpha92)
	__epi_1xf64 alpha93 = BROADCAST_f64( *(a11 + 9 + 3 * 16), gvl ); // (alpha93)
	__epi_1xf64 alpha94 = BROADCAST_f64( *(a11 + 9 + 4 * 16), gvl ); // (alpha94)
	__epi_1xf64 alpha95 = BROADCAST_f64( *(a11 + 9 + 5 * 16), gvl ); // (alpha95)
	__epi_1xf64 alpha96 = BROADCAST_f64( *(a11 + 9 + 6 * 16), gvl ); // (alpha96)
	__epi_1xf64 alpha97 = BROADCAST_f64( *(a11 + 9 + 7 * 16), gvl ); // (alpha97)
	__epi_1xf64 alpha98 = BROADCAST_f64( *(a11 + 9 + 8 * 16), gvl ); // (alpha98)
	__epi_1xf64 alpha99 = BROADCAST_f64( *(a11 + 9 + 9 * 16), gvl ); // (1/alpha99)

	accum = __builtin_epi_vfmul_1xf64( cv00, alpha90, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha91, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha92, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha93, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha94, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha95, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha96, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha97, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha98, cv80, gvl );    // 

	cv90 = __builtin_epi_vfsub_1xf64( cv90, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv90 = __builtin_epi_vfmul_1xf64( cv90, alpha99, gvl ); // abv0 *= alpha00
#else
	cv90 = __builtin_epi_vfdiv_1xf64( cv90, alpha99, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 9*nr + 0*cs_c+0*vlen, cv90, gvl ); // Store row 10 of b11


	//
	// Iteration 10 --------------------------------------------
	//
	__epi_1xf64 alphaa0 = BROADCAST_f64( *(a11 + 10 + 0 * 16), gvl ); // (alphaa0)
	__epi_1xf64 alphaa1 = BROADCAST_f64( *(a11 + 10 + 1 * 16), gvl ); // (alphaa1)
	__epi_1xf64 alphaa2 = BROADCAST_f64( *(a11 + 10 + 2 * 16), gvl ); // (alphaa2)
	__epi_1xf64 alphaa3 = BROADCAST_f64( *(a11 + 10 + 3 * 16), gvl ); // (alphaa3)
	__epi_1xf64 alphaa4 = BROADCAST_f64( *(a11 + 10 + 4 * 16), gvl ); // (alphaa4)
	__epi_1xf64 alphaa5 = BROADCAST_f64( *(a11 + 10 + 5 * 16), gvl ); // (alphaa5)
	__epi_1xf64 alphaa6 = BROADCAST_f64( *(a11 + 10 + 6 * 16), gvl ); // (alphaa6)
	__epi_1xf64 alphaa7 = BROADCAST_f64( *(a11 + 10 + 7 * 16), gvl ); // (alphaa7)
	__epi_1xf64 alphaa8 = BROADCAST_f64( *(a11 + 10 + 8 * 16), gvl ); // (alphaa8)
	__epi_1xf64 alphaa9 = BROADCAST_f64( *(a11 + 10 + 9 * 16), gvl ); // (alphaa9)
	__epi_1xf64 alphaaa = BROADCAST_f64( *(a11 + 10 + 10 * 16), gvl ); // (1/alphaaa)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphaa0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaa9, cv90, gvl );    // 

	cva0 = __builtin_epi_vfsub_1xf64( cva0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cva0 = __builtin_epi_vfmul_1xf64( cva0, alphaaa, gvl ); // abv0 *= alpha00
#else
	cva0 = __builtin_epi_vfdiv_1xf64( cva0, alphaaa, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 10*nr + 0*cs_c+0*vlen, cva0, gvl ); // Store row 11 of b11


	//
	// Iteration 11 --------------------------------------------
	//
	__epi_1xf64 alphab0 = BROADCAST_f64( *(a11 + 11 + 0 * 16), gvl ); // (alphab0)
	__epi_1xf64 alphab1 = BROADCAST_f64( *(a11 + 11 + 1 * 16), gvl ); // (alphab1)
	__epi_1xf64 alphab2 = BROADCAST_f64( *(a11 + 11 + 2 * 16), gvl ); // (alphab2)
	__epi_1xf64 alphab3 = BROADCAST_f64( *(a11 + 11 + 3 * 16), gvl ); // (alphab3)
	__epi_1xf64 alphab4 = BROADCAST_f64( *(a11 + 11 + 4 * 16), gvl ); // (alphab4)
	__epi_1xf64 alphab5 = BROADCAST_f64( *(a11 + 11 + 5 * 16), gvl ); // (alphab5)
	__epi_1xf64 alphab6 = BROADCAST_f64( *(a11 + 11 + 6 * 16), gvl ); // (alphab6)
	__epi_1xf64 alphab7 = BROADCAST_f64( *(a11 + 11 + 7 * 16), gvl ); // (alphab7)
	__epi_1xf64 alphab8 = BROADCAST_f64( *(a11 + 11 + 8 * 16), gvl ); // (alphab8)
	__epi_1xf64 alphab9 = BROADCAST_f64( *(a11 + 11 + 9 * 16), gvl ); // (alphab9)
	__epi_1xf64 alphaba = BROADCAST_f64( *(a11 + 11 + 10 * 16), gvl ); // (alphaba)
	__epi_1xf64 alphabb = BROADCAST_f64( *(a11 + 11 + 11 * 16), gvl ); // (1/alphabb)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphab0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphab9, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaba, cva0, gvl );    // 

	cvb0 = __builtin_epi_vfsub_1xf64( cvb0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvb0 = __builtin_epi_vfmul_1xf64( cvb0, alphabb, gvl ); // abv0 *= alpha00
#else
	cvb0 = __builtin_epi_vfdiv_1xf64( cvb0, alphabb, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 11*nr + 0*cs_c+0*vlen, cvb0, gvl ); // Store row 12 of b11


	//
	// Iteration 12 --------------------------------------------
	//
	__epi_1xf64 alphac0 = BROADCAST_f64( *(a11 + 12 + 0 * 16), gvl ); // (alphac0)
	__epi_1xf64 alphac1 = BROADCAST_f64( *(a11 + 12 + 1 * 16), gvl ); // (alphac1)
	__epi_1xf64 alphac2 = BROADCAST_f64( *(a11 + 12 + 2 * 16), gvl ); // (alphac2)
	__epi_1xf64 alphac3 = BROADCAST_f64( *(a11 + 12 + 3 * 16), gvl ); // (alphac3)
	__epi_1xf64 alphac4 = BROADCAST_f64( *(a11 + 12 + 4 * 16), gvl ); // (alphac4)
	__epi_1xf64 alphac5 = BROADCAST_f64( *(a11 + 12 + 5 * 16), gvl ); // (alphac5)
	__epi_1xf64 alphac6 = BROADCAST_f64( *(a11 + 12 + 6 * 16), gvl ); // (alphac6)
	__epi_1xf64 alphac7 = BROADCAST_f64( *(a11 + 12 + 7 * 16), gvl ); // (alphac7)
	__epi_1xf64 alphac8 = BROADCAST_f64( *(a11 + 12 + 8 * 16), gvl ); // (alphac8)
	__epi_1xf64 alphac9 = BROADCAST_f64( *(a11 + 12 + 9 * 16), gvl ); // (alphac9)
	__epi_1xf64 alphaca = BROADCAST_f64( *(a11 + 12 + 10 * 16), gvl ); // (alphaca)
	__epi_1xf64 alphacb = BROADCAST_f64( *(a11 + 12 + 11 * 16), gvl ); // (alphacb)
	__epi_1xf64 alphacc = BROADCAST_f64( *(a11 + 12 + 12 * 16), gvl ); // (1/alphacc)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphac0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphac9, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaca, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphacb, cvb0, gvl );    // 

	cvc0 = __builtin_epi_vfsub_1xf64( cvc0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvc0 = __builtin_epi_vfmul_1xf64( cvc0, alphacc, gvl ); // abv0 *= alpha00
#else
	cvc0 = __builtin_epi_vfdiv_1xf64( cvc0, alphacc, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 12*nr + 0*cs_c+0*vlen, cvc0, gvl ); // Store row 13 of b11


	//
	// Iteration 13 --------------------------------------------
	//
	__epi_1xf64 alphad0 = BROADCAST_f64( *(a11 + 13 + 0 * 16), gvl ); // (alphad0)
	__epi_1xf64 alphad1 = BROADCAST_f64( *(a11 + 13 + 1 * 16), gvl ); // (alphad1)
	__epi_1xf64 alphad2 = BROADCAST_f64( *(a11 + 13 + 2 * 16), gvl ); // (alphad2)
	__epi_1xf64 alphad3 = BROADCAST_f64( *(a11 + 13 + 3 * 16), gvl ); // (alphad3)
	__epi_1xf64 alphad4 = BROADCAST_f64( *(a11 + 13 + 4 * 16), gvl ); // (alphad4)
	__epi_1xf64 alphad5 = BROADCAST_f64( *(a11 + 13 + 5 * 16), gvl ); // (alphad5)
	__epi_1xf64 alphad6 = BROADCAST_f64( *(a11 + 13 + 6 * 16), gvl ); // (alphad6)
	__epi_1xf64 alphad7 = BROADCAST_f64( *(a11 + 13 + 7 * 16), gvl ); // (alphad7)
	__epi_1xf64 alphad8 = BROADCAST_f64( *(a11 + 13 + 8 * 16), gvl ); // (alphad8)
	__epi_1xf64 alphad9 = BROADCAST_f64( *(a11 + 13 + 9 * 16), gvl ); // (alphad9)
	__epi_1xf64 alphada = BROADCAST_f64( *(a11 + 13 + 10 * 16), gvl ); // (alphada)
	__epi_1xf64 alphadb = BROADCAST_f64( *(a11 + 13 + 11 * 16), gvl ); // (alphadb)
	__epi_1xf64 alphadc = BROADCAST_f64( *(a11 + 13 + 12 * 16), gvl ); // (alphadc)
	__epi_1xf64 alphadd = BROADCAST_f64( *(a11 + 13 + 13 * 16), gvl ); // (1/alphadd)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphad0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphad9, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphada, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphadb, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphadc, cvc0, gvl );    // 

	cvd0 = __builtin_epi_vfsub_1xf64( cvd0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvd0 = __builtin_epi_vfmul_1xf64( cvd0, alphadd, gvl ); // abv0 *= alpha00
#else
	cvd0 = __builtin_epi_vfdiv_1xf64( cvd0, alphadd, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 13*nr + 0*cs_c+0*vlen, cvd0, gvl ); // Store row 14 of b11


	//
	// Iteration 14 --------------------------------------------
	//
	__epi_1xf64 alphae0 = BROADCAST_f64( *(a11 + 14 + 0 * 16), gvl ); // (alphae0)
	__epi_1xf64 alphae1 = BROADCAST_f64( *(a11 + 14 + 1 * 16), gvl ); // (alphae1)
	__epi_1xf64 alphae2 = BROADCAST_f64( *(a11 + 14 + 2 * 16), gvl ); // (alphae2)
	__epi_1xf64 alphae3 = BROADCAST_f64( *(a11 + 14 + 3 * 16), gvl ); // (alphae3)
	__epi_1xf64 alphae4 = BROADCAST_f64( *(a11 + 14 + 4 * 16), gvl ); // (alphae4)
	__epi_1xf64 alphae5 = BROADCAST_f64( *(a11 + 14 + 5 * 16), gvl ); // (alphae5)
	__epi_1xf64 alphae6 = BROADCAST_f64( *(a11 + 14 + 6 * 16), gvl ); // (alphae6)
	__epi_1xf64 alphae7 = BROADCAST_f64( *(a11 + 14 + 7 * 16), gvl ); // (alphae7)
	__epi_1xf64 alphae8 = BROADCAST_f64( *(a11 + 14 + 8 * 16), gvl ); // (alphae8)
	__epi_1xf64 alphae9 = BROADCAST_f64( *(a11 + 14 + 9 * 16), gvl ); // (alphae9)
	__epi_1xf64 alphaea = BROADCAST_f64( *(a11 + 14 + 10 * 16), gvl ); // (alphaea)
	__epi_1xf64 alphaeb = BROADCAST_f64( *(a11 + 14 + 11 * 16), gvl ); // (alphaeb)
	__epi_1xf64 alphaec = BROADCAST_f64( *(a11 + 14 + 12 * 16), gvl ); // (alphaec)
	__epi_1xf64 alphaed = BROADCAST_f64( *(a11 + 14 + 13 * 16), gvl ); // (alphaed)
	__epi_1xf64 alphaee = BROADCAST_f64( *(a11 + 14 + 14 * 16), gvl ); // (1/alphaee)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphae0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphae9, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaea, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaeb, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaec, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaed, cvd0, gvl );    // 

	cve0 = __builtin_epi_vfsub_1xf64( cve0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cve0 = __builtin_epi_vfmul_1xf64( cve0, alphaee, gvl ); // abv0 *= alpha00
#else
	cve0 = __builtin_epi_vfdiv_1xf64( cve0, alphaee, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 14*nr + 0*cs_c+0*vlen, cve0, gvl ); // Store row 15 of b11


	//
	// Iteration 15 --------------------------------------------
	//
	__epi_1xf64 alphaf0 = BROADCAST_f64( *(a11 + 15 + 0 * 16), gvl ); // (alphaf0)
	__epi_1xf64 alphaf1 = BROADCAST_f64( *(a11 + 15 + 1 * 16), gvl ); // (alphaf1)
	__epi_1xf64 alphaf2 = BROADCAST_f64( *(a11 + 15 + 2 * 16), gvl ); // (alphaf2)
	__epi_1xf64 alphaf3 = BROADCAST_f64( *(a11 + 15 + 3 * 16), gvl ); // (alphaf3)
	__epi_1xf64 alphaf4 = BROADCAST_f64( *(a11 + 15 + 4 * 16), gvl ); // (alphaf4)
	__epi_1xf64 alphaf5 = BROADCAST_f64( *(a11 + 15 + 5 * 16), gvl ); // (alphaf5)
	__epi_1xf64 alphaf6 = BROADCAST_f64( *(a11 + 15 + 6 * 16), gvl ); // (alphaf6)
	__epi_1xf64 alphaf7 = BROADCAST_f64( *(a11 + 15 + 7 * 16), gvl ); // (alphaf7)
	__epi_1xf64 alphaf8 = BROADCAST_f64( *(a11 + 15 + 8 * 16), gvl ); // (alphaf8)
	__epi_1xf64 alphaf9 = BROADCAST_f64( *(a11 + 15 + 9 * 16), gvl ); // (alphaf9)
	__epi_1xf64 alphafa = BROADCAST_f64( *(a11 + 15 + 10 * 16), gvl ); // (alphafa)
	__epi_1xf64 alphafb = BROADCAST_f64( *(a11 + 15 + 11 * 16), gvl ); // (alphafb)
	__epi_1xf64 alphafc = BROADCAST_f64( *(a11 + 15 + 12 * 16), gvl ); // (alphafc)
	__epi_1xf64 alphafd = BROADCAST_f64( *(a11 + 15 + 13 * 16), gvl ); // (alphafd)
	__epi_1xf64 alphafe = BROADCAST_f64( *(a11 + 15 + 14 * 16), gvl ); // (alphafe)
	__epi_1xf64 alphaff = BROADCAST_f64( *(a11 + 15 + 15 * 16), gvl ); // (1/alphaff)

	accum = __builtin_epi_vfmul_1xf64( cv00, alphaf0, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf1, cv10, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf2, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf3, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf4, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf5, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf6, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf7, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf8, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaf9, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphafa, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphafb, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphafc, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphafd, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphafe, cve0, gvl );    // 

	cvf0 = __builtin_epi_vfsub_1xf64( cvf0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvf0 = __builtin_epi_vfmul_1xf64( cvf0, alphaff, gvl ); // abv0 *= alpha00
#else
	cvf0 = __builtin_epi_vfdiv_1xf64( cvf0, alphaff, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 15*nr + 0*cs_c+0*vlen, cvf0, gvl ); // Store row 16 of b11


	// END TRSM

	// Row-major.
	if( cs_c == 1 )
	{
		// Store row 0
		__builtin_epi_vstore_1xf64( c11 + 0*rs_c + 0*cs_c+0*vlen,    cv00, gvl );
		// Store row 1
		__builtin_epi_vstore_1xf64( c11 + 1*rs_c + 0*cs_c+0*vlen,    cv10, gvl );
		// Store row 2
		__builtin_epi_vstore_1xf64( c11 + 2*rs_c + 0*cs_c+0*vlen,    cv20, gvl );
		// Store row 3
		__builtin_epi_vstore_1xf64( c11 + 3*rs_c + 0*cs_c+0*vlen,    cv30, gvl );
		// Store row 4
		__builtin_epi_vstore_1xf64( c11 + 4*rs_c + 0*cs_c+0*vlen,    cv40, gvl );
		// Store row 5
		__builtin_epi_vstore_1xf64( c11 + 5*rs_c + 0*cs_c+0*vlen,    cv50, gvl );
		// Store row 6
		__builtin_epi_vstore_1xf64( c11 + 6*rs_c + 0*cs_c+0*vlen,    cv60, gvl );
		// Store row 7
		__builtin_epi_vstore_1xf64( c11 + 7*rs_c + 0*cs_c+0*vlen,    cv70, gvl );
                // Store row 8
		__builtin_epi_vstore_1xf64( c11 + 8*rs_c + 0*cs_c+0*vlen,    cv80, gvl );
		// Store row 9
		__builtin_epi_vstore_1xf64( c11 + 9*rs_c + 0*cs_c+0*vlen,    cv90, gvl );
		// Store row 10
		__builtin_epi_vstore_1xf64( c11 + 10*rs_c + 0*cs_c+0*vlen,    cva0, gvl );
		// Store row 11
		__builtin_epi_vstore_1xf64( c11 + 11*rs_c + 0*cs_c+0*vlen,    cvb0, gvl );
		// Store row 12
		__builtin_epi_vstore_1xf64( c11 + 12*rs_c + 0*cs_c+0*vlen,    cvc0, gvl );
		// Store row 13
		__builtin_epi_vstore_1xf64( c11 + 13*rs_c + 0*cs_c+0*vlen,    cvd0, gvl );
		// Store row 14
		__builtin_epi_vstore_1xf64( c11 + 14*rs_c + 0*cs_c+0*vlen,    cve0, gvl );
		// Store row 15
		__builtin_epi_vstore_1xf64( c11 + 15*rs_c + 0*cs_c+0*vlen,    cvf0, gvl );

	}
#if 1
	// Column-major.
	if( rs_c == 1 )
	{
		// Store row 0
		__builtin_epi_vstore_strided_1xf64( c11 + 0*rs_c + 0*cs_c+0*vlen,    cv00, (long int)cs_c*sizeof(double) , gvl );
		// Store row 1
		__builtin_epi_vstore_strided_1xf64( c11 + 1*rs_c + 0*cs_c+0*vlen,    cv10, (long int)cs_c*sizeof(double) , gvl );
		// Store row 2
		__builtin_epi_vstore_strided_1xf64( c11 + 2*rs_c + 0*cs_c+0*vlen,    cv20, (long int)cs_c*sizeof(double) , gvl );
		// Store row 3
		__builtin_epi_vstore_strided_1xf64( c11 + 3*rs_c + 0*cs_c+0*vlen,    cv30, (long int)cs_c*sizeof(double) , gvl );
		// Store row 4
		__builtin_epi_vstore_strided_1xf64( c11 + 4*rs_c + 0*cs_c+0*vlen,    cv40, (long int)cs_c*sizeof(double) , gvl );
		// Store row 5
		__builtin_epi_vstore_strided_1xf64( c11 + 5*rs_c + 0*cs_c+0*vlen,    cv50, (long int)cs_c*sizeof(double) , gvl );
		// Store row 6
		__builtin_epi_vstore_strided_1xf64( c11 + 6*rs_c + 0*cs_c+0*vlen,    cv60, (long int)cs_c*sizeof(double) , gvl );
		// Store row 7
		__builtin_epi_vstore_strided_1xf64( c11 + 7*rs_c + 0*cs_c+0*vlen,    cv70, (long int)cs_c*sizeof(double) , gvl );
                // Store row 8
		__builtin_epi_vstore_strided_1xf64( c11 + 8*rs_c + 0*cs_c+0*vlen,    cv80, (long int)cs_c*sizeof(double) , gvl );
		// Store row 9
		__builtin_epi_vstore_strided_1xf64( c11 + 9*rs_c + 0*cs_c+0*vlen,    cv90, (long int)cs_c*sizeof(double) , gvl );
		// Store row 10
		__builtin_epi_vstore_strided_1xf64( c11 + 10*rs_c + 0*cs_c+0*vlen,    cva0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 11
		__builtin_epi_vstore_strided_1xf64( c11 + 11*rs_c + 0*cs_c+0*vlen,    cvb0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 12
		__builtin_epi_vstore_strided_1xf64( c11 + 12*rs_c + 0*cs_c+0*vlen,    cvc0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 13
		__builtin_epi_vstore_strided_1xf64( c11 + 13*rs_c + 0*cs_c+0*vlen,    cvd0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 14
		__builtin_epi_vstore_strided_1xf64( c11 + 14*rs_c + 0*cs_c+0*vlen,    cve0, (long int)cs_c*sizeof(double) , gvl );
		// Store row 15
		__builtin_epi_vstore_strided_1xf64( c11 + 15*rs_c + 0*cs_c+0*vlen,    cvf0, (long int)cs_c*sizeof(double) , gvl );

	}
#endif

	GEMMTRSM_UKR_FLUSH_CT( d );
}
