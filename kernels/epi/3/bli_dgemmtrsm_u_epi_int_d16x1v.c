#include "blis.h"

#ifdef EPI_FPGA
#define BROADCAST_f64 __builtin_epi_vbroadcast_1xf64
#else
#define BROADCAST_f64 __builtin_epi_vfmv_v_f_1xf64
#endif

void bli_dgemmtrsm_u_epi_scalar_16x1v
     (
       dim_t                m,
       dim_t                n,
       dim_t                k0,
       double*     restrict alpha,
       double*     restrict a12,
       double*     restrict a11,
       double*     restrict b21,
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

	//long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );
	//long gvl = __builtin_epi_vsetvl( mr, __epi_e64, __epi_m1 );
	long gvl = __builtin_epi_vsetvl( nr, __epi_e64, __epi_m1 );

        unsigned long int vlen = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);

	//printf("GEMMTRSM MICROKERNEL. M: %d, N: %d, K: %d - cs_c: %d, rs_c: %d - alpha: %f\n", m, n, k0, cs_c, rs_c, *alpha);

	__epi_1xf64 alphav;
	alphav = BROADCAST_f64( *alpha, gvl );

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
	abv00 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 1)
	abv10 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 2)
	abv20 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 3)
	abv30 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 4)
	abv40 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 5)
	abv50 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 6)
	abv60 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 7)
	abv70 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 8)
	abv80 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 9)
	abv90 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 10)
	abva0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 11)
	abvb0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 12)
	abvc0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 13)
	abvd0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 14)
	abve0 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (column 15)
	abvf0 = BROADCAST_f64( 0.0, gvl );

	__epi_1xf64 sav1;

	//printf("k=%d %d\n", k0, k_iter);
	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iteration 0
 		bv00 = __builtin_epi_vload_1xf64( b21+0*vlen, gvl );

 		sav1 = BROADCAST_f64( *(a12), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+1), gvl );
		abv10 = __builtin_epi_vfmacc_1xf64( abv10, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+2), gvl );
		abv20 = __builtin_epi_vfmacc_1xf64( abv20, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+3), gvl );
		abv30 = __builtin_epi_vfmacc_1xf64( abv30, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+4), gvl );
		abv40 = __builtin_epi_vfmacc_1xf64( abv40, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+5), gvl );
		abv50 = __builtin_epi_vfmacc_1xf64( abv50, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+6), gvl );
		abv60 = __builtin_epi_vfmacc_1xf64( abv60, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+7), gvl );
		abv70 = __builtin_epi_vfmacc_1xf64( abv70, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+8), gvl );
		abv80 = __builtin_epi_vfmacc_1xf64( abv80, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+9), gvl );
		abv90 = __builtin_epi_vfmacc_1xf64( abv90, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+10), gvl );
		abva0 = __builtin_epi_vfmacc_1xf64( abva0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+11), gvl );
		abvb0 = __builtin_epi_vfmacc_1xf64( abvb0, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+12), gvl );
		abvc0 = __builtin_epi_vfmacc_1xf64( abvc0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+13), gvl );
		abvd0 = __builtin_epi_vfmacc_1xf64( abvd0, sav1, bv00, gvl );

 		sav1 = BROADCAST_f64( *(a12+14), gvl );
		abve0 = __builtin_epi_vfmacc_1xf64( abve0, sav1, bv00, gvl );

		sav1 = BROADCAST_f64( *(a12+15), gvl );
		abvf0 = __builtin_epi_vfmacc_1xf64( abvf0, sav1, bv00, gvl );

	        // Adjust pointers for next iterations.
	        //b21 += 1 * vlen * 1; // 1 vectors + 1 Unroll factor
	        b21 += nr; // 1 vectors + 1 Unroll factor
		//a12 += 16;
		a12 += mr;
	}
	
	//printf("Done loop\n");

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

	// b11 := alpha * b11 - a12 * b21
	//
	// Load alpha and duplicate
	//__epi_1xf64 alphav;
	//alphav = BROADCAST_f64( *alpha, gvl );

	// b11 := alpha * b11 - a12 * b21
 	cv00 = __builtin_epi_vload_1xf64( b11+0*nr, gvl );
	cv00 = __builtin_epi_vfmsub_1xf64( cv00, alphav, abv00, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+0*nr, cv00, gvl );

 	cv10 = __builtin_epi_vload_1xf64( b11+1*nr, gvl );
	cv10 = __builtin_epi_vfmsub_1xf64( cv10, alphav, abv10, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+1*nr, cv10, gvl );
	
 	cv20 = __builtin_epi_vload_1xf64( b11+2*nr, gvl );
	cv20 = __builtin_epi_vfmsub_1xf64( cv20, alphav, abv20, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+2*nr, cv20, gvl );

 	cv30 = __builtin_epi_vload_1xf64( b11+3*nr, gvl );
	cv30 = __builtin_epi_vfmsub_1xf64( cv30, alphav, abv30, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+3*nr, cv30, gvl );

 	cv40 = __builtin_epi_vload_1xf64( b11+4*nr, gvl );
	cv40 = __builtin_epi_vfmsub_1xf64( cv40, alphav, abv40, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+4*nr, cv40, gvl );

 	cv50 = __builtin_epi_vload_1xf64( b11+5*nr, gvl );
	cv50 = __builtin_epi_vfmsub_1xf64( cv50, alphav, abv50, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+5*nr, cv50, gvl );

 	cv60 = __builtin_epi_vload_1xf64( b11+6*nr, gvl );
	cv60 = __builtin_epi_vfmsub_1xf64( cv60, alphav, abv60, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+6*nr, cv60, gvl );

 	cv70 = __builtin_epi_vload_1xf64( b11+7*nr, gvl );
	cv70 = __builtin_epi_vfmsub_1xf64( cv70, alphav, abv70, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+7*nr, cv70, gvl );

 	cv80 = __builtin_epi_vload_1xf64( b11+8*nr, gvl );
	cv80 = __builtin_epi_vfmsub_1xf64( cv80, alphav, abv80, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+8*nr, cv80, gvl );

 	cv90 = __builtin_epi_vload_1xf64( b11+9*nr, gvl );
	cv90 = __builtin_epi_vfmsub_1xf64( cv90, alphav, abv90, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+9*nr, cv90, gvl );

 	cva0 = __builtin_epi_vload_1xf64( b11+10*nr, gvl );
	cva0 = __builtin_epi_vfmsub_1xf64( cva0, alphav, abva0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+10*nr, cva0, gvl );

 	cvb0 = __builtin_epi_vload_1xf64( b11+11*nr, gvl );
	cvb0 = __builtin_epi_vfmsub_1xf64( cvb0, alphav, abvb0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+11*nr, cvb0, gvl );

 	cvc0 = __builtin_epi_vload_1xf64( b11+12*nr, gvl );
	cvc0 = __builtin_epi_vfmsub_1xf64( cvc0, alphav, abvc0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+12*nr, cvc0, gvl );

 	cvd0 = __builtin_epi_vload_1xf64( b11+13*nr, gvl );
	cvd0 = __builtin_epi_vfmsub_1xf64( cvd0, alphav, abvd0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+13*nr, cvd0, gvl );

 	cve0 = __builtin_epi_vload_1xf64( b11+14*nr, gvl );
	cve0 = __builtin_epi_vfmsub_1xf64( cve0, alphav, abve0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+14*nr, cve0, gvl );

 	cvf0 = __builtin_epi_vload_1xf64( b11+15*nr, gvl );
	cvf0 = __builtin_epi_vfmsub_1xf64( cvf0, alphav, abvf0, gvl );
 	       //__builtin_epi_vstore_1xf64( b11+15*nr, cvf0, gvl );

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
	__epi_1xf64 alphaff = BROADCAST_f64( *(a11 + 15 + 15 * 16), gvl ); // (alphaff)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvf0 = __builtin_epi_vfmul_1xf64( cvf0, alphaff, gvl ); // cvf0 *= (1/alphaff)
#else
	cvf0 = __builtin_epi_vfdiv_1xf64( cvf0, alphaff, gvl ); // cvf0 /= alphaff
#endif

	__builtin_epi_vstore_1xf64( b11 + 15*nr + 0*cs_c+0*vlen, cvf0, gvl ); // Store row 15 of b11

	//
	// Iteration 1  --------------------------------------------
	//
	__epi_1xf64 alphaef = BROADCAST_f64( *(a11 + 14 + 15 * 16), gvl ); // (alphaef)
	__epi_1xf64 alphaee = BROADCAST_f64( *(a11 + 14 + 14 * 16), gvl ); // (alphaee)

	__epi_1xf64 accum = __builtin_epi_vfmul_1xf64( cvf0, alphaef, gvl );  // Scale row 1 of B11

	cve0  = __builtin_epi_vfsub_1xf64( cve0, accum, gvl );    // Update row 14 of B11

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cve0 = __builtin_epi_vfmul_1xf64( cve0, alphaee, gvl ); // abv0 *= (1/alphaee)
#else
	cve0 = __builtin_epi_vfdiv_1xf64( cve0, alphaee, gvl ); // abv0 /= alphaee
#endif

	__builtin_epi_vstore_1xf64( b11 + 14*nr + 0*cs_c+0*vlen, cve0, gvl ); // Store row 14 of b11

	//
	// Iteration 2  --------------------------------------------
	//
	__epi_1xf64 alphadf = BROADCAST_f64( *(a11 + 13 + 15 * 16), gvl ); // (alphadf)
	__epi_1xf64 alphade = BROADCAST_f64( *(a11 + 13 + 14 * 16), gvl ); // (alphade)
	__epi_1xf64 alphadd = BROADCAST_f64( *(a11 + 13 + 13 * 16), gvl ); // (alphadd)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alphadf, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphade, cve0, gvl );    // 

	cvd0 = __builtin_epi_vfsub_1xf64( cvd0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvd0 = __builtin_epi_vfmul_1xf64( cvd0, alphadd, gvl ); // abv0 *= alpha00
#else
	cvd0 = __builtin_epi_vfdiv_1xf64( cvd0, alphadd, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 13*nr + 0*cs_c+0*vlen, cvd0, gvl ); // Store row 13 of b11

	//
	// Iteration 3  --------------------------------------------
	//
	__epi_1xf64 alphacf = BROADCAST_f64( *(a11 + 12 + 15 * 16), gvl ); // (alphacf)
	__epi_1xf64 alphace = BROADCAST_f64( *(a11 + 12 + 14 * 16), gvl ); // (alphace)
	__epi_1xf64 alphacd = BROADCAST_f64( *(a11 + 12 + 13 * 16), gvl ); // (alphacd)
	__epi_1xf64 alphacc = BROADCAST_f64( *(a11 + 12 + 12 * 16), gvl ); // (1/alphacc)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alphacf, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphace, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphacd, cvd0, gvl );    // 

	cvc0 = __builtin_epi_vfsub_1xf64( cvc0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvc0 = __builtin_epi_vfmul_1xf64( cvc0, alphacc, gvl ); // abv0 *= alpha00
#else
	cvc0 = __builtin_epi_vfdiv_1xf64( cvc0, alphacc, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 12*nr + 0*cs_c+0*vlen, cvc0, gvl ); // Store row 4 of b11


	//
	// Iteration 4  --------------------------------------------
	//
	__epi_1xf64 alphabf = BROADCAST_f64( *(a11 + 11 + 15 * 16), gvl ); // (alphabf)
	__epi_1xf64 alphabe = BROADCAST_f64( *(a11 + 11 + 14 * 16), gvl ); // (alphabe)
	__epi_1xf64 alphabd = BROADCAST_f64( *(a11 + 11 + 13 * 16), gvl ); // (alphabd)
	__epi_1xf64 alphabc = BROADCAST_f64( *(a11 + 11 + 12 * 16), gvl ); // (alphabc)
	__epi_1xf64 alphabb = BROADCAST_f64( *(a11 + 11 + 11 * 16), gvl ); // (1/alphabb)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alphabf, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphabe, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphabd, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphabc, cvc0, gvl );    // 

	cvb0 = __builtin_epi_vfsub_1xf64( cvb0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cvb0 = __builtin_epi_vfmul_1xf64( cvb0, alphabb, gvl ); // abv0 *= alpha00
#else
	cvb0 = __builtin_epi_vfdiv_1xf64( cvb0, alphabb, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 11*nr + 0*cs_c+0*vlen, cvb0, gvl ); // Store row 5 of b11


	//
	// Iteration 5  --------------------------------------------
	//
	__epi_1xf64 alphaaf = BROADCAST_f64( *(a11 + 10 + 15 * 16), gvl ); // (alphaaf)
	__epi_1xf64 alphaae = BROADCAST_f64( *(a11 + 10 + 14 * 16), gvl ); // (alphaae)
	__epi_1xf64 alphaad = BROADCAST_f64( *(a11 + 10 + 13 * 16), gvl ); // (alphaad)
	__epi_1xf64 alphaac = BROADCAST_f64( *(a11 + 10 + 12 * 16), gvl ); // (alphaac)
	__epi_1xf64 alphaab = BROADCAST_f64( *(a11 + 10 + 11 * 16), gvl ); // (alphaab)
	__epi_1xf64 alphaaa = BROADCAST_f64( *(a11 + 10 + 10 * 16), gvl ); // (1/alphaaa)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alphaaf, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaae, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaad, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaac, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alphaab, cvb0, gvl );    // 

	cva0 = __builtin_epi_vfsub_1xf64( cva0, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cva0 = __builtin_epi_vfmul_1xf64( cva0, alphaaa, gvl ); // abv0 *= alpha00
#else
	cva0 = __builtin_epi_vfdiv_1xf64( cva0, alphaaa, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 10*nr + 0*cs_c+0*vlen, cva0, gvl ); // Store row 6 of b11


	//
	// Iteration 6  --------------------------------------------
	//
	__epi_1xf64 alpha9f = BROADCAST_f64( *(a11 + 9 + 15 * 16), gvl ); // (alpha9f)
	__epi_1xf64 alpha9e = BROADCAST_f64( *(a11 + 9 + 14 * 16), gvl ); // (alpha9e)
	__epi_1xf64 alpha9d = BROADCAST_f64( *(a11 + 9 + 13 * 16), gvl ); // (alpha9d)
	__epi_1xf64 alpha9c = BROADCAST_f64( *(a11 + 9 + 12 * 16), gvl ); // (alpha9c)
	__epi_1xf64 alpha9b = BROADCAST_f64( *(a11 + 9 + 11 * 16), gvl ); // (alpha9b)
	__epi_1xf64 alpha9a = BROADCAST_f64( *(a11 + 9 + 10 * 16), gvl ); // (alpha9a)
	__epi_1xf64 alpha99 = BROADCAST_f64( *(a11 + 9 + 9 * 16), gvl ); // (1/alpha99)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha9f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha9e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha9d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha9c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha9b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha9a, cva0, gvl );    // 

	cv90 = __builtin_epi_vfsub_1xf64( cv90, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv90 = __builtin_epi_vfmul_1xf64( cv90, alpha99, gvl ); // abv0 *= alpha00
#else
	cv90 = __builtin_epi_vfdiv_1xf64( cv90, alpha99, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 9*nr + 0*cs_c+0*vlen, cv90, gvl ); // Store row 7 of b11



	//
	// Iteration 7  --------------------------------------------
	//
	__epi_1xf64 alpha8f = BROADCAST_f64( *(a11 + 8 + 15 * 16), gvl ); // (alpha8f)
	__epi_1xf64 alpha8e = BROADCAST_f64( *(a11 + 8 + 14 * 16), gvl ); // (alpha8e)
	__epi_1xf64 alpha8d = BROADCAST_f64( *(a11 + 8 + 13 * 16), gvl ); // (alpha8d)
	__epi_1xf64 alpha8c = BROADCAST_f64( *(a11 + 8 + 12 * 16), gvl ); // (alpha8c)
	__epi_1xf64 alpha8b = BROADCAST_f64( *(a11 + 8 + 11 * 16), gvl ); // (alpha8b)
	__epi_1xf64 alpha8a = BROADCAST_f64( *(a11 + 8 + 10 * 16), gvl ); // (alpha8a)
	__epi_1xf64 alpha89 = BROADCAST_f64( *(a11 + 8 + 9 * 16), gvl ); // (alpha89)
	__epi_1xf64 alpha88 = BROADCAST_f64( *(a11 + 8 + 8 * 16), gvl ); // (1/alpha88)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha8f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha8e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha8d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha8c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha8b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha8a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha89, cv90, gvl );    // 

	cv80 = __builtin_epi_vfsub_1xf64( cv80, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv80 = __builtin_epi_vfmul_1xf64( cv80, alpha88, gvl ); // abv0 *= alpha00
#else
	cv80 = __builtin_epi_vfdiv_1xf64( cv80, alpha88, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 8*nr + 0*cs_c+0*vlen, cv80, gvl ); // Store row 8 of b11


	//
	// Iteration 8  --------------------------------------------
	//
	__epi_1xf64 alpha7f = BROADCAST_f64( *(a11 + 7 + 15 * 16), gvl ); // (alpha7f)
	__epi_1xf64 alpha7e = BROADCAST_f64( *(a11 + 7 + 14 * 16), gvl ); // (alpha7e)
	__epi_1xf64 alpha7d = BROADCAST_f64( *(a11 + 7 + 13 * 16), gvl ); // (alpha7d)
	__epi_1xf64 alpha7c = BROADCAST_f64( *(a11 + 7 + 12 * 16), gvl ); // (alpha7c)
	__epi_1xf64 alpha7b = BROADCAST_f64( *(a11 + 7 + 11 * 16), gvl ); // (alpha7b)
	__epi_1xf64 alpha7a = BROADCAST_f64( *(a11 + 7 + 10 * 16), gvl ); // (alpha7a)
	__epi_1xf64 alpha79 = BROADCAST_f64( *(a11 + 7 + 9 * 16), gvl ); // (alpha79)
	__epi_1xf64 alpha78 = BROADCAST_f64( *(a11 + 7 + 8 * 16), gvl ); // (alpha78)
	__epi_1xf64 alpha77 = BROADCAST_f64( *(a11 + 7 + 7 * 16), gvl ); // (1/alpha77)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha7f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha7e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha7d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha7c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha7b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha7a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha79, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha78, cv80, gvl );    // 

	cv70 = __builtin_epi_vfsub_1xf64( cv70, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv70 = __builtin_epi_vfmul_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
#else
	cv70 = __builtin_epi_vfdiv_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 7*nr + 0*cs_c+0*vlen, cv70, gvl ); // Store row 9 of b11


	//
	// Iteration 9  --------------------------------------------
	//
	__epi_1xf64 alpha6f = BROADCAST_f64( *(a11 + 6 + 15 * 16), gvl ); // (alpha6f)
	__epi_1xf64 alpha6e = BROADCAST_f64( *(a11 + 6 + 14 * 16), gvl ); // (alpha6e)
	__epi_1xf64 alpha6d = BROADCAST_f64( *(a11 + 6 + 13 * 16), gvl ); // (alpha6d)
	__epi_1xf64 alpha6c = BROADCAST_f64( *(a11 + 6 + 12 * 16), gvl ); // (alpha6c)
	__epi_1xf64 alpha6b = BROADCAST_f64( *(a11 + 6 + 11 * 16), gvl ); // (alpha6b)
	__epi_1xf64 alpha6a = BROADCAST_f64( *(a11 + 6 + 10 * 16), gvl ); // (alpha6a)
	__epi_1xf64 alpha69 = BROADCAST_f64( *(a11 + 6 + 9 * 16), gvl ); // (alpha69)
	__epi_1xf64 alpha68 = BROADCAST_f64( *(a11 + 6 + 8 * 16), gvl ); // (alpha68)
	__epi_1xf64 alpha67 = BROADCAST_f64( *(a11 + 6 + 7 * 16), gvl ); // (alpha67)
	__epi_1xf64 alpha66 = BROADCAST_f64( *(a11 + 6 + 6 * 16), gvl ); // (1/alpha66)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha6f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha6e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha6d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha6c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha6b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha6a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha69, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha68, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha67, cv70, gvl );    // 

	cv60 = __builtin_epi_vfsub_1xf64( cv60, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv60 = __builtin_epi_vfmul_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
#else
	cv60 = __builtin_epi_vfdiv_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 6*nr + 0*cs_c+0*vlen, cv60, gvl ); // Store row 10 of b11


	//
	// Iteration 10 --------------------------------------------
	//
	__epi_1xf64 alpha5f = BROADCAST_f64( *(a11 + 5 + 15 * 16), gvl ); // (alpha5f)
	__epi_1xf64 alpha5e = BROADCAST_f64( *(a11 + 5 + 14 * 16), gvl ); // (alpha5e)
	__epi_1xf64 alpha5d = BROADCAST_f64( *(a11 + 5 + 13 * 16), gvl ); // (alpha5d)
	__epi_1xf64 alpha5c = BROADCAST_f64( *(a11 + 5 + 12 * 16), gvl ); // (alpha5c)
	__epi_1xf64 alpha5b = BROADCAST_f64( *(a11 + 5 + 11 * 16), gvl ); // (alpha5b)
	__epi_1xf64 alpha5a = BROADCAST_f64( *(a11 + 5 + 10 * 16), gvl ); // (alpha5a)
	__epi_1xf64 alpha59 = BROADCAST_f64( *(a11 + 5 + 9 * 16), gvl ); // (alpha59)
	__epi_1xf64 alpha58 = BROADCAST_f64( *(a11 + 5 + 8 * 16), gvl ); // (alpha58)
	__epi_1xf64 alpha57 = BROADCAST_f64( *(a11 + 5 + 7 * 16), gvl ); // (alpha57)
	__epi_1xf64 alpha56 = BROADCAST_f64( *(a11 + 5 + 6 * 16), gvl ); // (alpha56)
	__epi_1xf64 alpha55 = BROADCAST_f64( *(a11 + 5 + 5 * 16), gvl ); // (1/alpha55)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha5f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha5e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha5d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha5c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha5b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha5a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha59, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha58, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha57, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha56, cv60, gvl );    // 

	cv50 = __builtin_epi_vfsub_1xf64( cv50, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv50 = __builtin_epi_vfmul_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
#else
	cv50 = __builtin_epi_vfdiv_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 5*nr + 0*cs_c+0*vlen, cv50, gvl ); // Store row 11 of b11


	//
	// Iteration 11 --------------------------------------------
	//
	__epi_1xf64 alpha4f = BROADCAST_f64( *(a11 + 4 + 15 * 16), gvl ); // (alpha4f)
	__epi_1xf64 alpha4e = BROADCAST_f64( *(a11 + 4 + 14 * 16), gvl ); // (alpha4e)
	__epi_1xf64 alpha4d = BROADCAST_f64( *(a11 + 4 + 13 * 16), gvl ); // (alpha4d)
	__epi_1xf64 alpha4c = BROADCAST_f64( *(a11 + 4 + 12 * 16), gvl ); // (alpha4c)
	__epi_1xf64 alpha4b = BROADCAST_f64( *(a11 + 4 + 11 * 16), gvl ); // (alpha4b)
	__epi_1xf64 alpha4a = BROADCAST_f64( *(a11 + 4 + 10 * 16), gvl ); // (alpha4a)
	__epi_1xf64 alpha49 = BROADCAST_f64( *(a11 + 4 + 9 * 16), gvl ); // (alpha49)
	__epi_1xf64 alpha48 = BROADCAST_f64( *(a11 + 4 + 8 * 16), gvl ); // (alpha48)
	__epi_1xf64 alpha47 = BROADCAST_f64( *(a11 + 4 + 7 * 16), gvl ); // (alpha47)
	__epi_1xf64 alpha46 = BROADCAST_f64( *(a11 + 4 + 6 * 16), gvl ); // (alpha46)
	__epi_1xf64 alpha45 = BROADCAST_f64( *(a11 + 4 + 5 * 16), gvl ); // (alpha45)
	__epi_1xf64 alpha44 = BROADCAST_f64( *(a11 + 4 + 4 * 16), gvl ); // (1/alpha44)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha4f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha4e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha4d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha4c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha4b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha4a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha49, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha48, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha47, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha46, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha45, cv50, gvl );    // 

	cv40 = __builtin_epi_vfsub_1xf64( cv40, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv40 = __builtin_epi_vfmul_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
#else
	cv40 = __builtin_epi_vfdiv_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 4*nr + 0*cs_c+0*vlen, cv40, gvl ); // Store row 12 of b11


	//
	// Iteration 12 --------------------------------------------
	//
	__epi_1xf64 alpha3f = BROADCAST_f64( *(a11 + 3 + 15 * 16), gvl ); // (alpha3f)
	__epi_1xf64 alpha3e = BROADCAST_f64( *(a11 + 3 + 14 * 16), gvl ); // (alpha3e)
	__epi_1xf64 alpha3d = BROADCAST_f64( *(a11 + 3 + 13 * 16), gvl ); // (alpha3d)
	__epi_1xf64 alpha3c = BROADCAST_f64( *(a11 + 3 + 12 * 16), gvl ); // (alpha3c)
	__epi_1xf64 alpha3b = BROADCAST_f64( *(a11 + 3 + 11 * 16), gvl ); // (alpha3b)
	__epi_1xf64 alpha3a = BROADCAST_f64( *(a11 + 3 + 10 * 16), gvl ); // (alpha3a)
	__epi_1xf64 alpha39 = BROADCAST_f64( *(a11 + 3 + 9 * 16), gvl ); // (alpha39)
	__epi_1xf64 alpha38 = BROADCAST_f64( *(a11 + 3 + 8 * 16), gvl ); // (alpha38)
	__epi_1xf64 alpha37 = BROADCAST_f64( *(a11 + 3 + 7 * 16), gvl ); // (alpha37)
	__epi_1xf64 alpha36 = BROADCAST_f64( *(a11 + 3 + 6 * 16), gvl ); // (alpha36)
	__epi_1xf64 alpha35 = BROADCAST_f64( *(a11 + 3 + 5 * 16), gvl ); // (alpha35)
	__epi_1xf64 alpha34 = BROADCAST_f64( *(a11 + 3 + 4 * 16), gvl ); // (alpha34)
	__epi_1xf64 alpha33 = BROADCAST_f64( *(a11 + 3 + 3 * 16), gvl ); // (1/alpha33)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha3f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha3e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha3d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha3c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha3b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha3a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha39, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha38, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha37, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha36, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha35, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha34, cv40, gvl );    // 

	cv30 = __builtin_epi_vfsub_1xf64( cv30, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv30 = __builtin_epi_vfmul_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
#else
	cv30 = __builtin_epi_vfdiv_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 3*nr + 0*cs_c+0*vlen, cv30, gvl ); // Store row 13 of b11


	//
	// Iteration 13 --------------------------------------------
	//
	__epi_1xf64 alpha2f = BROADCAST_f64( *(a11 + 2 + 15 * 16), gvl ); // (alpha2f)
	__epi_1xf64 alpha2e = BROADCAST_f64( *(a11 + 2 + 14 * 16), gvl ); // (alpha2e)
	__epi_1xf64 alpha2d = BROADCAST_f64( *(a11 + 2 + 13 * 16), gvl ); // (alpha2d)
	__epi_1xf64 alpha2c = BROADCAST_f64( *(a11 + 2 + 12 * 16), gvl ); // (alpha2c)
	__epi_1xf64 alpha2b = BROADCAST_f64( *(a11 + 2 + 11 * 16), gvl ); // (alpha2b)
	__epi_1xf64 alpha2a = BROADCAST_f64( *(a11 + 2 + 10 * 16), gvl ); // (alpha2a)
	__epi_1xf64 alpha29 = BROADCAST_f64( *(a11 + 2 + 9 * 16), gvl ); // (alpha29)
	__epi_1xf64 alpha28 = BROADCAST_f64( *(a11 + 2 + 8 * 16), gvl ); // (alpha28)
	__epi_1xf64 alpha27 = BROADCAST_f64( *(a11 + 2 + 7 * 16), gvl ); // (alpha27)
	__epi_1xf64 alpha26 = BROADCAST_f64( *(a11 + 2 + 6 * 16), gvl ); // (alpha26)
	__epi_1xf64 alpha25 = BROADCAST_f64( *(a11 + 2 + 5 * 16), gvl ); // (alpha25)
	__epi_1xf64 alpha24 = BROADCAST_f64( *(a11 + 2 + 4 * 16), gvl ); // (alpha24)
	__epi_1xf64 alpha23 = BROADCAST_f64( *(a11 + 2 + 3 * 16), gvl ); // (alpha23)
	__epi_1xf64 alpha22 = BROADCAST_f64( *(a11 + 2 + 2 * 16), gvl ); // (1/alpha22)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha2f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha2e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha2d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha2c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha2b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha2a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha29, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha28, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha27, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha26, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha25, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha24, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha23, cv30, gvl );    // 

	cv20 = __builtin_epi_vfsub_1xf64( cv20, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv20 = __builtin_epi_vfmul_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
#else
	cv20 = __builtin_epi_vfdiv_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 2*nr + 0*cs_c+0*vlen, cv20, gvl ); // Store row 14 of b11


	//
	// Iteration 14 --------------------------------------------
	//
	__epi_1xf64 alpha1f = BROADCAST_f64( *(a11 + 1 + 15 * 16), gvl ); // (alpha1f)
	__epi_1xf64 alpha1e = BROADCAST_f64( *(a11 + 1 + 14 * 16), gvl ); // (alpha1e)
	__epi_1xf64 alpha1d = BROADCAST_f64( *(a11 + 1 + 13 * 16), gvl ); // (alpha1d)
	__epi_1xf64 alpha1c = BROADCAST_f64( *(a11 + 1 + 12 * 16), gvl ); // (alpha1c)
	__epi_1xf64 alpha1b = BROADCAST_f64( *(a11 + 1 + 11 * 16), gvl ); // (alpha1b)
	__epi_1xf64 alpha1a = BROADCAST_f64( *(a11 + 1 + 10 * 16), gvl ); // (alpha1a)
	__epi_1xf64 alpha19 = BROADCAST_f64( *(a11 + 1 + 9 * 16), gvl ); // (alpha19)
	__epi_1xf64 alpha18 = BROADCAST_f64( *(a11 + 1 + 8 * 16), gvl ); // (alpha18)
	__epi_1xf64 alpha17 = BROADCAST_f64( *(a11 + 1 + 7 * 16), gvl ); // (alpha17)
	__epi_1xf64 alpha16 = BROADCAST_f64( *(a11 + 1 + 6 * 16), gvl ); // (alpha16)
	__epi_1xf64 alpha15 = BROADCAST_f64( *(a11 + 1 + 5 * 16), gvl ); // (alpha15)
	__epi_1xf64 alpha14 = BROADCAST_f64( *(a11 + 1 + 4 * 16), gvl ); // (alpha14)
	__epi_1xf64 alpha13 = BROADCAST_f64( *(a11 + 1 + 3 * 16), gvl ); // (alpha13)
	__epi_1xf64 alpha12 = BROADCAST_f64( *(a11 + 1 + 2 * 16), gvl ); // (alpha12)
	__epi_1xf64 alpha11 = BROADCAST_f64( *(a11 + 1 + 1 * 16), gvl ); // (1/alpha11)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha1f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha1e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha1d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha1c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha1b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha1a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha19, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha18, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha17, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha16, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha15, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha14, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha13, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha12, cv20, gvl );    // 

	cv10 = __builtin_epi_vfsub_1xf64( cv10, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv10 = __builtin_epi_vfmul_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
#else
	cv10 = __builtin_epi_vfdiv_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 1*nr + 0*cs_c+0*vlen, cv10, gvl ); // Store row 15 of b11


	//
	// Iteration 15 --------------------------------------------
	//
	__epi_1xf64 alpha0f = BROADCAST_f64( *(a11 + 0 + 15 * 16), gvl ); // (alpha0f)
	__epi_1xf64 alpha0e = BROADCAST_f64( *(a11 + 0 + 14 * 16), gvl ); // (alpha0e)
	__epi_1xf64 alpha0d = BROADCAST_f64( *(a11 + 0 + 13 * 16), gvl ); // (alpha0d)
	__epi_1xf64 alpha0c = BROADCAST_f64( *(a11 + 0 + 12 * 16), gvl ); // (alpha0c)
	__epi_1xf64 alpha0b = BROADCAST_f64( *(a11 + 0 + 11 * 16), gvl ); // (alpha0b)
	__epi_1xf64 alpha0a = BROADCAST_f64( *(a11 + 0 + 10 * 16), gvl ); // (alpha0a)
	__epi_1xf64 alpha09 = BROADCAST_f64( *(a11 + 0 + 9 * 16), gvl ); // (alpha09)
	__epi_1xf64 alpha08 = BROADCAST_f64( *(a11 + 0 + 8 * 16), gvl ); // (alpha08)
	__epi_1xf64 alpha07 = BROADCAST_f64( *(a11 + 0 + 7 * 16), gvl ); // (alpha07)
	__epi_1xf64 alpha06 = BROADCAST_f64( *(a11 + 0 + 6 * 16), gvl ); // (alpha06)
	__epi_1xf64 alpha05 = BROADCAST_f64( *(a11 + 0 + 5 * 16), gvl ); // (alpha05)
	__epi_1xf64 alpha04 = BROADCAST_f64( *(a11 + 0 + 4 * 16), gvl ); // (alpha04)
	__epi_1xf64 alpha03 = BROADCAST_f64( *(a11 + 0 + 3 * 16), gvl ); // (alpha03)
	__epi_1xf64 alpha02 = BROADCAST_f64( *(a11 + 0 + 2 * 16), gvl ); // (alpha02)
	__epi_1xf64 alpha01 = BROADCAST_f64( *(a11 + 0 + 1 * 16), gvl ); // (alpha01)
	__epi_1xf64 alpha00 = BROADCAST_f64( *(a11 + 0 + 0 * 16), gvl ); // (1/alpha00)

	accum = __builtin_epi_vfmul_1xf64( cvf0, alpha0f, gvl );            // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha0e, cve0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha0d, cvd0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha0c, cvc0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha0b, cvb0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha0a, cva0, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha09, cv90, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha08, cv80, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha07, cv70, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha06, cv60, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha05, cv50, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha04, cv40, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha03, cv30, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha02, cv20, gvl );    // 
	accum = __builtin_epi_vfmacc_1xf64( accum, alpha01, cv10, gvl );    // 

	cv00 = __builtin_epi_vfsub_1xf64( cv00, accum, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv00 = __builtin_epi_vfmul_1xf64( cv00, alpha00, gvl ); // abv0 *= alpha00
#else
	cv00 = __builtin_epi_vfdiv_1xf64( cv00, alpha00, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 0*nr + 0*cs_c+0*vlen, cv00, gvl ); // Store row 16 of b11


	// END TRSM
        ////printf("FINISHING: rs_c = %d, cs_c = %d\n", rs_c, cs_c);

	// Row-major.
	if( cs_c == 1 )
	{
		////printf("WAY 1: cs_c == 1; rs_c = %d\n", rs_c);
		// Store row 0
		__builtin_epi_vstore_1xf64( c11 + 0*rs_c + 0*cs_c+0*vlen,    cv00, gvl );
	        //printf("C00 post: %le\n", *(c11));
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
		////printf("done cs_c == 1\n");

	}
#if 1
	// Column-major.
	if( rs_c == 1 )
	{
		//printf("WAY 2: rs_c == 1, cs_c = %d\n", cs_c);
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
		//printf("done rs_c == 1\n");

	}
#endif
	//printf("Ended gemmtrsm: cs_c == %d; rs_c = %d\n", cs_c, rs_c);

	GEMMTRSM_UKR_FLUSH_CT( d );
}
