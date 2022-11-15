#include "blis.h"

#ifdef EPI_FPGA
#define BROADCAST_f64 __builtin_epi_vbroadcast_1xf64
#else
#define BROADCAST_f64 __builtin_epi_vfmv_v_f_1xf64
#endif

void bli_dgemmtrsm_l_epi_scalar_8x3v
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

	GEMMTRSM_UKR_SETUP_CT( d, bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ),
                                  bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NR, cntx ),
				  true );

	long gvl = __builtin_epi_vsetvl( nr/3, __epi_e64, __epi_m1 );
        unsigned long int vlen = 240;

	// B vectors.
	__epi_1xf64 bv00, bv01, bv02;

	// C row (3v).
	__epi_1xf64 cv00, cv01, cv02,
	            cv10, cv11, cv12,
	            cv20, cv21, cv22,
	            cv30, cv31, cv32,
	            cv40, cv41, cv42,
	            cv50, cv51, cv52,
	            cv60, cv61, cv62,
	            cv70, cv71, cv72;

	// Accummulators (8x3).
	__epi_1xf64 abv00, abv01, abv02, 
		    abv10, abv11, abv12, 
		    abv20, abv21, abv22, 
		    abv30, abv31, abv32, 
		    abv40, abv41, abv42, 
		    abv50, abv51, abv52, 
		    abv60, abv61, abv62, 
		    abv70, abv71, abv72;

	// Initialize accummulators to 0.0 (row 0)
	abv00 = BROADCAST_f64( 0.0, gvl );
	abv01 = BROADCAST_f64( 0.0, gvl );
	abv02 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 1)
	abv10 = BROADCAST_f64( 0.0, gvl );
	abv11 = BROADCAST_f64( 0.0, gvl );
	abv12 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 2)
	abv20 = BROADCAST_f64( 0.0, gvl );
	abv21 = BROADCAST_f64( 0.0, gvl );
	abv22 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 3)
	abv30 = BROADCAST_f64( 0.0, gvl );
	abv31 = BROADCAST_f64( 0.0, gvl );
	abv32 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 4)
	abv40 = BROADCAST_f64( 0.0, gvl );
	abv41 = BROADCAST_f64( 0.0, gvl );
	abv42 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 5)
	abv50 = BROADCAST_f64( 0.0, gvl );
	abv51 = BROADCAST_f64( 0.0, gvl );
	abv52 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 6)
	abv60 = BROADCAST_f64( 0.0, gvl );
	abv61 = BROADCAST_f64( 0.0, gvl );
	abv62 = BROADCAST_f64( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 7)
	abv70 = BROADCAST_f64( 0.0, gvl );
	abv71 = BROADCAST_f64( 0.0, gvl );
	abv72 = BROADCAST_f64( 0.0, gvl );

	__epi_1xf64 sav1;

	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iteration 0
 		bv00 = __builtin_epi_vload_1xf64( b01+0*vlen, gvl );
 		bv01 = __builtin_epi_vload_1xf64( b01+1*vlen, gvl );
 		bv02 = __builtin_epi_vload_1xf64( b01+2*vlen, gvl );

 		sav1 = BROADCAST_f64( *(a10), gvl );
		abv00 = __builtin_epi_vfmacc_1xf64( abv00, sav1, bv00, gvl );
		abv01 = __builtin_epi_vfmacc_1xf64( abv01, sav1, bv01, gvl );
		abv02 = __builtin_epi_vfmacc_1xf64( abv02, sav1, bv02, gvl );

		sav1 = BROADCAST_f64( *(a10+1), gvl );
		abv10 = __builtin_epi_vfmacc_1xf64( abv10, sav1, bv00, gvl );
		abv11 = __builtin_epi_vfmacc_1xf64( abv11, sav1, bv01, gvl );
		abv12 = __builtin_epi_vfmacc_1xf64( abv12, sav1, bv02, gvl );

 		sav1 = BROADCAST_f64( *(a10+2), gvl );
		abv20 = __builtin_epi_vfmacc_1xf64( abv20, sav1, bv00, gvl );
		abv21 = __builtin_epi_vfmacc_1xf64( abv21, sav1, bv01, gvl );
		abv22 = __builtin_epi_vfmacc_1xf64( abv22, sav1, bv02, gvl );

		sav1 = BROADCAST_f64( *(a10+3), gvl );
		abv30 = __builtin_epi_vfmacc_1xf64( abv30, sav1, bv00, gvl );
		abv31 = __builtin_epi_vfmacc_1xf64( abv31, sav1, bv01, gvl );
		abv32 = __builtin_epi_vfmacc_1xf64( abv32, sav1, bv02, gvl );

 		sav1 = BROADCAST_f64( *(a10+4), gvl );
		abv40 = __builtin_epi_vfmacc_1xf64( abv40, sav1, bv00, gvl );
		abv41 = __builtin_epi_vfmacc_1xf64( abv41, sav1, bv01, gvl );
		abv42 = __builtin_epi_vfmacc_1xf64( abv42, sav1, bv02, gvl );

		sav1 = BROADCAST_f64( *(a10+5), gvl );
		abv50 = __builtin_epi_vfmacc_1xf64( abv50, sav1, bv00, gvl );
		abv51 = __builtin_epi_vfmacc_1xf64( abv51, sav1, bv01, gvl );
		abv52 = __builtin_epi_vfmacc_1xf64( abv52, sav1, bv02, gvl );

 		sav1 = BROADCAST_f64( *(a10+6), gvl );
		abv60 = __builtin_epi_vfmacc_1xf64( abv60, sav1, bv00, gvl );
		abv61 = __builtin_epi_vfmacc_1xf64( abv61, sav1, bv01, gvl );
		abv62 = __builtin_epi_vfmacc_1xf64( abv62, sav1, bv02, gvl );

		sav1 = BROADCAST_f64( *(a10+7), gvl );
		abv70 = __builtin_epi_vfmacc_1xf64( abv70, sav1, bv00, gvl );
		abv71 = __builtin_epi_vfmacc_1xf64( abv71, sav1, bv01, gvl );
		abv72 = __builtin_epi_vfmacc_1xf64( abv72, sav1, bv02, gvl );

 		// Adjust pointers for next iterations.
	        //b01 += 3 * vlen * 1; // 3 vectors + 1 Unroll factor
		//a10 += 8;
	        b01 += nr;
		a10 += mr;
	}


	// b11 := alpha * b11 - a10 * b01
	//
	// Load alpha and duplicate
	__epi_1xf64 alphav;
	alphav = BROADCAST_f64( *alpha, gvl );

	// b11 := alpha * b11 - a10 * b01

	// ROW of b11
 	cv00 = __builtin_epi_vload_1xf64( b11 + 0 *nr+cs_c*0*vlen, gvl );
	cv00 = __builtin_epi_vfmsub_1xf64( cv00, alphav, abv00, gvl );
	//
 	cv01 = __builtin_epi_vload_1xf64( b11 + 0 *nr+cs_c*1*vlen, gvl );
	cv01 = __builtin_epi_vfmsub_1xf64( cv01, alphav, abv01, gvl );
	//
 	cv02 = __builtin_epi_vload_1xf64( b11 + 0 *nr+cs_c*2*vlen, gvl );
	cv02 = __builtin_epi_vfmsub_1xf64( cv02, alphav, abv02, gvl );

	// ROW of b11
 	cv10 = __builtin_epi_vload_1xf64( b11 + 1 *nr+cs_c*0*vlen, gvl );
	cv10 = __builtin_epi_vfmsub_1xf64( cv10, alphav, abv10, gvl );
	//
 	cv11 = __builtin_epi_vload_1xf64( b11 + 1 *nr+cs_c*1*vlen, gvl );
	cv11 = __builtin_epi_vfmsub_1xf64( cv11, alphav, abv11, gvl );
	//
 	cv12 = __builtin_epi_vload_1xf64( b11 + 1 *nr+cs_c*2*vlen, gvl );
	cv12 = __builtin_epi_vfmsub_1xf64( cv12, alphav, abv12, gvl );

	// ROW of b11
 	cv20 = __builtin_epi_vload_1xf64( b11 + 2 *nr+cs_c*0*vlen, gvl );
	cv20 = __builtin_epi_vfmsub_1xf64( cv20, alphav, abv20, gvl );
	//
 	cv21 = __builtin_epi_vload_1xf64( b11 + 2 *nr+cs_c*1*vlen, gvl );
	cv21 = __builtin_epi_vfmsub_1xf64( cv21, alphav, abv21, gvl );
	//
 	cv22 = __builtin_epi_vload_1xf64( b11 + 2 *nr+cs_c*2*vlen, gvl );
	cv22 = __builtin_epi_vfmsub_1xf64( cv22, alphav, abv22, gvl );

	// ROW of b11
 	cv30 = __builtin_epi_vload_1xf64( b11 + 3 *nr+cs_c*0*vlen, gvl );
	cv30 = __builtin_epi_vfmsub_1xf64( cv30, alphav, abv30, gvl );
	//
 	cv31 = __builtin_epi_vload_1xf64( b11 + 3 *nr+cs_c*1*vlen, gvl );
	cv31 = __builtin_epi_vfmsub_1xf64( cv31, alphav, abv31, gvl );
	//
 	cv32 = __builtin_epi_vload_1xf64( b11 + 3 *nr+cs_c*2*vlen, gvl );
	cv32 = __builtin_epi_vfmsub_1xf64( cv32, alphav, abv32, gvl );

	// ROW of b11
 	cv40 = __builtin_epi_vload_1xf64( b11 + 4 *nr+cs_c*0*vlen, gvl );
	cv40 = __builtin_epi_vfmsub_1xf64( cv40, alphav, abv40, gvl );
	//
 	cv41 = __builtin_epi_vload_1xf64( b11 + 4 *nr+cs_c*1*vlen, gvl );
	cv41 = __builtin_epi_vfmsub_1xf64( cv41, alphav, abv41, gvl );
	//
 	cv42 = __builtin_epi_vload_1xf64( b11 + 4 *nr+cs_c*2*vlen, gvl );
	cv42 = __builtin_epi_vfmsub_1xf64( cv42, alphav, abv42, gvl );

	// ROW of b11
 	cv50 = __builtin_epi_vload_1xf64( b11 + 5 *nr+cs_c*0*vlen, gvl );
	cv50 = __builtin_epi_vfmsub_1xf64( cv50, alphav, abv50, gvl );
	//
 	cv51 = __builtin_epi_vload_1xf64( b11 + 5 *nr+cs_c*1*vlen, gvl );
	cv51 = __builtin_epi_vfmsub_1xf64( cv51, alphav, abv51, gvl );
	//
 	cv52 = __builtin_epi_vload_1xf64( b11 + 5 *nr+cs_c*2*vlen, gvl );
	cv52 = __builtin_epi_vfmsub_1xf64( cv52, alphav, abv52, gvl );

	// ROW of b11
 	cv60 = __builtin_epi_vload_1xf64( b11 + 6 *nr+cs_c*0*vlen, gvl );
	cv60 = __builtin_epi_vfmsub_1xf64( cv60, alphav, abv60, gvl );
	//
 	cv61 = __builtin_epi_vload_1xf64( b11 + 6 *nr+cs_c*1*vlen, gvl );
	cv61 = __builtin_epi_vfmsub_1xf64( cv61, alphav, abv61, gvl );
	//
 	cv62 = __builtin_epi_vload_1xf64( b11 + 6 *nr+cs_c*2*vlen, gvl );
	cv62 = __builtin_epi_vfmsub_1xf64( cv62, alphav, abv62, gvl );

	// ROW of b11
 	cv70 = __builtin_epi_vload_1xf64( b11 + 7 *nr+cs_c*0*vlen, gvl );
	cv70 = __builtin_epi_vfmsub_1xf64( cv70, alphav, abv70, gvl );
	//
 	cv71 = __builtin_epi_vload_1xf64( b11 + 7 *nr+cs_c*1*vlen, gvl );
	cv71 = __builtin_epi_vfmsub_1xf64( cv71, alphav, abv71, gvl );
	//
 	cv72 = __builtin_epi_vload_1xf64( b11 + 7 *nr+cs_c*2*vlen, gvl );
	cv72 = __builtin_epi_vfmsub_1xf64( cv72, alphav, abv72, gvl );

	// Contents of b11 are stored by rows (8 rows, from cv00 to cv72).
	// (cv00)  (cv01)  (cv02)
	// (cv10)  (cv11)  (cv12)
	// (cv20)  (cv21)  (cv22)
	// (cv30)  (cv31)  (cv32)
	// (cv40)  (cv41)  (cv42)
	// (cv50)  (cv51)  (cv52)
	// (cv60)  (cv61)  (cv62)
	// (cv70)  (cv71)  (cv72)
	//
	// Iteration 0  --------------------------------------------
	// HERE
 	__epi_1xf64 alpha00 = BROADCAST_f64( *(a11), gvl ); // (1/alpha00)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv00 = __builtin_epi_vfmul_1xf64( cv00, alpha00, gvl ); // cv00 *= (1/alpha00)
	cv01 = __builtin_epi_vfmul_1xf64( cv01, alpha00, gvl ); // cv00 *= (1/alpha00)
	cv02 = __builtin_epi_vfmul_1xf64( cv02, alpha00, gvl ); // cv00 *= (1/alpha00)
#else
	cv00 = __builtin_epi_vfdiv_1xf64( cv00, alpha00, gvl ); // cv00 /= alpha00
	cv01 = __builtin_epi_vfdiv_1xf64( cv01, alpha00, gvl ); // cv00 /= alpha00
	cv02 = __builtin_epi_vfdiv_1xf64( cv02, alpha00, gvl ); // cv00 /= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 0 * nr + 0 * vlen * cs_c, cv00, gvl ); // Store row 1 of b11
	__builtin_epi_vstore_1xf64( b11 + 0 * nr + 1 * vlen * cs_c, cv01, gvl ); // Store row 1 of b11
	__builtin_epi_vstore_1xf64( b11 + 0 * nr + 2 * vlen * cs_c, cv02, gvl ); // Store row 1 of b11

	//__builtin_epi_vstore_1xf64( b11 + 0*vlen + 0*cs_c+0*vlen, cv00, gvl ); // Store row 1 of b11
	//__builtin_epi_vstore_1xf64( b11 + 1*vlen + 0*cs_c+0*vlen, cv01, gvl ); // Store row 1 of b11
	//__builtin_epi_vstore_1xf64( b11 + 2*vlen + 0*cs_c+0*vlen, cv02, gvl ); // Store row 1 of b11


	//
	// Iteration 1  --------------------------------------------
	//
	__epi_1xf64 alpha10 = BROADCAST_f64( *(a11 + 1 + 0 * 8), gvl ); // (alpha10)
	__epi_1xf64 alpha11 = BROADCAST_f64( *(a11 + 1 + 1 * 8), gvl ); // (1/alpha11)

	__epi_1xf64 accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha10, gvl );  // Scale row 1 of B11
	__epi_1xf64 accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha10, gvl );  // Scale row 1 of B11
	__epi_1xf64 accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha10, gvl );  // Scale row 1 of B11

	cv10  = __builtin_epi_vfsub_1xf64( cv10, accum0, gvl );    // Update row 2 of B11
	cv11  = __builtin_epi_vfsub_1xf64( cv11, accum1, gvl );    // Update row 2 of B11
	cv12  = __builtin_epi_vfsub_1xf64( cv12, accum2, gvl );    // Update row 2 of B11

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv10 = __builtin_epi_vfmul_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
	cv11 = __builtin_epi_vfmul_1xf64( cv11, alpha11, gvl ); // abv0 *= alpha00
	cv12 = __builtin_epi_vfmul_1xf64( cv12, alpha11, gvl ); // abv0 *= alpha00
#else
	cv10 = __builtin_epi_vfdiv_1xf64( cv10, alpha11, gvl ); // abv0 *= alpha00
	cv11 = __builtin_epi_vfdiv_1xf64( cv11, alpha11, gvl ); // abv0 *= alpha00
	cv12 = __builtin_epi_vfdiv_1xf64( cv12, alpha11, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 1 * nr + 0 * vlen * cs_c, cv10, gvl ); // Store row 2 of b11
	__builtin_epi_vstore_1xf64( b11 + 1 * nr + 1 * vlen * cs_c, cv11, gvl ); // Store row 2 of b11
	__builtin_epi_vstore_1xf64( b11 + 1 * nr + 2 * vlen * cs_c, cv12, gvl ); // Store row 2 of b11

	//__builtin_epi_vstore_1xf64( b11 + 3*vlen + 0*cs_c+0*vlen, cv10, gvl ); // Store row 2 of b11
	//__builtin_epi_vstore_1xf64( b11 + 4*vlen + 0*cs_c+0*vlen, cv11, gvl ); // Store row 2 of b11
	//__builtin_epi_vstore_1xf64( b11 + 5*vlen + 0*cs_c+0*vlen, cv12, gvl ); // Store row 2 of b11


	//
	// Iteration 2  --------------------------------------------
	//
	__epi_1xf64 alpha20 = BROADCAST_f64( *(a11 + 2 + 0 * 8), gvl ); // (alpha20)
	__epi_1xf64 alpha21 = BROADCAST_f64( *(a11 + 2 + 1 * 8), gvl ); // (alpha21)
	__epi_1xf64 alpha22 = BROADCAST_f64( *(a11 + 2 + 2 * 8), gvl ); // (1/alpha22)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha20, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha20, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha20, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha21, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha21, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha21, cv12, gvl );    // 

	cv20 = __builtin_epi_vfsub_1xf64( cv20, accum0, gvl );    // 
	cv21 = __builtin_epi_vfsub_1xf64( cv21, accum1, gvl );    // 
	cv22 = __builtin_epi_vfsub_1xf64( cv22, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv20 = __builtin_epi_vfmul_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
	cv21 = __builtin_epi_vfmul_1xf64( cv21, alpha22, gvl ); // abv0 *= alpha00
	cv22 = __builtin_epi_vfmul_1xf64( cv22, alpha22, gvl ); // abv0 *= alpha00
#else
	cv20 = __builtin_epi_vfdiv_1xf64( cv20, alpha22, gvl ); // abv0 *= alpha00
	cv21 = __builtin_epi_vfdiv_1xf64( cv21, alpha22, gvl ); // abv0 *= alpha00
	cv22 = __builtin_epi_vfdiv_1xf64( cv22, alpha22, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 2 * nr + 0 * vlen * cs_c, cv20, gvl ); // Store row 3 of b11
	__builtin_epi_vstore_1xf64( b11 + 2 * nr + 1 * vlen * cs_c, cv21, gvl ); // Store row 3 of b11
	__builtin_epi_vstore_1xf64( b11 + 2 * nr + 2 * vlen * cs_c, cv22, gvl ); // Store row 3 of b11

	//__builtin_epi_vstore_1xf64( b11 + 6*vlen + 0*cs_c+0*vlen, cv20, gvl ); // Store row 3 of b11
	//__builtin_epi_vstore_1xf64( b11 + 7*vlen + 0*cs_c+0*vlen, cv21, gvl ); // Store row 3 of b11
	//__builtin_epi_vstore_1xf64( b11 + 8*vlen + 0*cs_c+0*vlen, cv22, gvl ); // Store row 3 of b11

	//
	// Iteration 3  --------------------------------------------
	//
	__epi_1xf64 alpha30 = BROADCAST_f64( *(a11 + 3 + 0 * 8), gvl ); // (alpha30)
	__epi_1xf64 alpha31 = BROADCAST_f64( *(a11 + 3 + 1 * 8), gvl ); // (alpha31)
	__epi_1xf64 alpha32 = BROADCAST_f64( *(a11 + 3 + 2 * 8), gvl ); // (alpha32)
	__epi_1xf64 alpha33 = BROADCAST_f64( *(a11 + 3 + 3 * 8), gvl ); // (1/alpha33)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha30, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha30, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha30, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha31, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha31, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha31, cv12, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha32, cv20, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha32, cv21, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha32, cv22, gvl );    // 

	cv30 = __builtin_epi_vfsub_1xf64( cv30, accum0, gvl );    // 
	cv31 = __builtin_epi_vfsub_1xf64( cv31, accum1, gvl );    // 
	cv32 = __builtin_epi_vfsub_1xf64( cv32, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv30 = __builtin_epi_vfmul_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
	cv31 = __builtin_epi_vfmul_1xf64( cv31, alpha33, gvl ); // abv0 *= alpha00
	cv32 = __builtin_epi_vfmul_1xf64( cv32, alpha33, gvl ); // abv0 *= alpha00
#else
	cv30 = __builtin_epi_vfdiv_1xf64( cv30, alpha33, gvl ); // abv0 *= alpha00
	cv31 = __builtin_epi_vfdiv_1xf64( cv31, alpha33, gvl ); // abv0 *= alpha00
	cv32 = __builtin_epi_vfdiv_1xf64( cv32, alpha33, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 3 * nr + 0 * vlen * cs_c, cv30, gvl ); // Store row 4 of b11
	__builtin_epi_vstore_1xf64( b11 + 3 * nr + 1 * vlen * cs_c, cv31, gvl ); // Store row 4 of b11
	__builtin_epi_vstore_1xf64( b11 + 3 * nr + 2 * vlen * cs_c, cv32, gvl ); // Store row 4 of b11

	//__builtin_epi_vstore_1xf64( b11 + 9*vlen + 0*cs_c+0*vlen, cv30, gvl ); // Store row 4 of b11
	//__builtin_epi_vstore_1xf64( b11 + 10*vlen + 0*cs_c+0*vlen, cv31, gvl ); // Store row 4 of b11
	//__builtin_epi_vstore_1xf64( b11 + 11*vlen + 0*cs_c+0*vlen, cv32, gvl ); // Store row 4 of b11


	//
	// Iteration 4  --------------------------------------------
	//
	__epi_1xf64 alpha40 = BROADCAST_f64( *(a11 + 4 + 0 * 8), gvl ); // (alpha40)
	__epi_1xf64 alpha41 = BROADCAST_f64( *(a11 + 4 + 1 * 8), gvl ); // (alpha41)
	__epi_1xf64 alpha42 = BROADCAST_f64( *(a11 + 4 + 2 * 8), gvl ); // (alpha42)
	__epi_1xf64 alpha43 = BROADCAST_f64( *(a11 + 4 + 3 * 8), gvl ); // (alpha43)
	__epi_1xf64 alpha44 = BROADCAST_f64( *(a11 + 4 + 4 * 8), gvl ); // (1/alpha44)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha40, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha40, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha40, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha41, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha41, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha41, cv12, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha42, cv20, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha42, cv21, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha42, cv22, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha43, cv30, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha43, cv31, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha43, cv32, gvl );    // 

	cv40 = __builtin_epi_vfsub_1xf64( cv40, accum0, gvl );    // 
	cv41 = __builtin_epi_vfsub_1xf64( cv41, accum1, gvl );    // 
	cv42 = __builtin_epi_vfsub_1xf64( cv42, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv40 = __builtin_epi_vfmul_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
	cv41 = __builtin_epi_vfmul_1xf64( cv41, alpha44, gvl ); // abv0 *= alpha00
	cv42 = __builtin_epi_vfmul_1xf64( cv42, alpha44, gvl ); // abv0 *= alpha00
#else
	cv40 = __builtin_epi_vfdiv_1xf64( cv40, alpha44, gvl ); // abv0 *= alpha00
	cv41 = __builtin_epi_vfdiv_1xf64( cv41, alpha44, gvl ); // abv0 *= alpha00
	cv42 = __builtin_epi_vfdiv_1xf64( cv42, alpha44, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 4 * nr + 0 * vlen * cs_c, cv40, gvl ); // Store row 5 of b11
	__builtin_epi_vstore_1xf64( b11 + 4 * nr + 1 * vlen * cs_c, cv41, gvl ); // Store row 5 of b11
	__builtin_epi_vstore_1xf64( b11 + 4 * nr + 2 * vlen * cs_c, cv42, gvl ); // Store row 5 of b11

	//__builtin_epi_vstore_1xf64( b11 + 12*vlen + 0*cs_c+0*vlen, cv40, gvl ); // Store row 5 of b11
	//__builtin_epi_vstore_1xf64( b11 + 13*vlen + 0*cs_c+0*vlen, cv41, gvl ); // Store row 5 of b11
	//__builtin_epi_vstore_1xf64( b11 + 14*vlen + 0*cs_c+0*vlen, cv42, gvl ); // Store row 5 of b11


	//
	// Iteration 5  --------------------------------------------
	//
	__epi_1xf64 alpha50 = BROADCAST_f64( *(a11 + 5 + 0 * 8), gvl ); // (alpha50)
	__epi_1xf64 alpha51 = BROADCAST_f64( *(a11 + 5 + 1 * 8), gvl ); // (alpha51)
	__epi_1xf64 alpha52 = BROADCAST_f64( *(a11 + 5 + 2 * 8), gvl ); // (alpha52)
	__epi_1xf64 alpha53 = BROADCAST_f64( *(a11 + 5 + 3 * 8), gvl ); // (alpha53)
	__epi_1xf64 alpha54 = BROADCAST_f64( *(a11 + 5 + 4 * 8), gvl ); // (alpha54)
	__epi_1xf64 alpha55 = BROADCAST_f64( *(a11 + 5 + 5 * 8), gvl ); // (1/alpha55)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha50, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha50, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha50, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha51, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha51, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha51, cv12, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha52, cv20, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha52, cv21, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha52, cv22, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha53, cv30, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha53, cv31, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha53, cv32, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha54, cv40, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha54, cv41, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha54, cv42, gvl );    // 

	cv50 = __builtin_epi_vfsub_1xf64( cv50, accum0, gvl );    // 
	cv51 = __builtin_epi_vfsub_1xf64( cv51, accum1, gvl );    // 
	cv52 = __builtin_epi_vfsub_1xf64( cv52, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv50 = __builtin_epi_vfmul_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
	cv51 = __builtin_epi_vfmul_1xf64( cv51, alpha55, gvl ); // abv0 *= alpha00
	cv52 = __builtin_epi_vfmul_1xf64( cv52, alpha55, gvl ); // abv0 *= alpha00
#else
	cv50 = __builtin_epi_vfdiv_1xf64( cv50, alpha55, gvl ); // abv0 *= alpha00
	cv51 = __builtin_epi_vfdiv_1xf64( cv51, alpha55, gvl ); // abv0 *= alpha00
	cv52 = __builtin_epi_vfdiv_1xf64( cv52, alpha55, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 5 * nr + 0 * vlen * cs_c, cv50, gvl ); // Store row 6 of b11
	__builtin_epi_vstore_1xf64( b11 + 5 * nr + 1 * vlen * cs_c, cv51, gvl ); // Store row 6 of b11
	__builtin_epi_vstore_1xf64( b11 + 5 * nr + 2 * vlen * cs_c, cv52, gvl ); // Store row 6 of b11

	//__builtin_epi_vstore_1xf64( b11 + 15*vlen + 0*cs_c+0*vlen, cv50, gvl ); // Store row 6 of b11
	//__builtin_epi_vstore_1xf64( b11 + 16*vlen + 0*cs_c+0*vlen, cv51, gvl ); // Store row 6 of b11
	//__builtin_epi_vstore_1xf64( b11 + 17*vlen + 0*cs_c+0*vlen, cv52, gvl ); // Store row 6 of b11


	//
	// Iteration 6  --------------------------------------------
	//
	__epi_1xf64 alpha60 = BROADCAST_f64( *(a11 + 6 + 0 * 8), gvl ); // (alpha60)
	__epi_1xf64 alpha61 = BROADCAST_f64( *(a11 + 6 + 1 * 8), gvl ); // (alpha61)
	__epi_1xf64 alpha62 = BROADCAST_f64( *(a11 + 6 + 2 * 8), gvl ); // (alpha62)
	__epi_1xf64 alpha63 = BROADCAST_f64( *(a11 + 6 + 3 * 8), gvl ); // (alpha63)
	__epi_1xf64 alpha64 = BROADCAST_f64( *(a11 + 6 + 4 * 8), gvl ); // (alpha64)
	__epi_1xf64 alpha65 = BROADCAST_f64( *(a11 + 6 + 5 * 8), gvl ); // (alpha65)
	__epi_1xf64 alpha66 = BROADCAST_f64( *(a11 + 6 + 6 * 8), gvl ); // (1/alpha66)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha60, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha60, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha60, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha61, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha61, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha61, cv12, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha62, cv20, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha62, cv21, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha62, cv22, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha63, cv30, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha63, cv31, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha63, cv32, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha64, cv40, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha64, cv41, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha64, cv42, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha65, cv50, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha65, cv51, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha65, cv52, gvl );    // 

	cv60 = __builtin_epi_vfsub_1xf64( cv60, accum0, gvl );    // 
	cv61 = __builtin_epi_vfsub_1xf64( cv61, accum1, gvl );    // 
	cv62 = __builtin_epi_vfsub_1xf64( cv62, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv60 = __builtin_epi_vfmul_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
	cv61 = __builtin_epi_vfmul_1xf64( cv61, alpha66, gvl ); // abv0 *= alpha00
	cv62 = __builtin_epi_vfmul_1xf64( cv62, alpha66, gvl ); // abv0 *= alpha00
#else
	cv60 = __builtin_epi_vfdiv_1xf64( cv60, alpha66, gvl ); // abv0 *= alpha00
	cv61 = __builtin_epi_vfdiv_1xf64( cv61, alpha66, gvl ); // abv0 *= alpha00
	cv62 = __builtin_epi_vfdiv_1xf64( cv62, alpha66, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 6 * nr + 0 * vlen * cs_c, cv60, gvl ); // Store row 7 of b11
	__builtin_epi_vstore_1xf64( b11 + 6 * nr + 1 * vlen * cs_c, cv61, gvl ); // Store row 7 of b11
	__builtin_epi_vstore_1xf64( b11 + 6 * nr + 2 * vlen * cs_c, cv62, gvl ); // Store row 7 of b11

	//__builtin_epi_vstore_1xf64( b11 + 18*vlen + 0*cs_c+0*vlen, cv60, gvl ); // Store row 7 of b11
	//__builtin_epi_vstore_1xf64( b11 + 19*vlen + 0*cs_c+0*vlen, cv61, gvl ); // Store row 7 of b11
	//__builtin_epi_vstore_1xf64( b11 + 20*vlen + 0*cs_c+0*vlen, cv62, gvl ); // Store row 7 of b11



	//
	// Iteration 7  --------------------------------------------
	//
	__epi_1xf64 alpha70 = BROADCAST_f64( *(a11 + 7 + 0 * 8), gvl ); // (alpha70)
	__epi_1xf64 alpha71 = BROADCAST_f64( *(a11 + 7 + 1 * 8), gvl ); // (alpha71)
	__epi_1xf64 alpha72 = BROADCAST_f64( *(a11 + 7 + 2 * 8), gvl ); // (alpha72)
	__epi_1xf64 alpha73 = BROADCAST_f64( *(a11 + 7 + 3 * 8), gvl ); // (alpha73)
	__epi_1xf64 alpha74 = BROADCAST_f64( *(a11 + 7 + 4 * 8), gvl ); // (alpha74)
	__epi_1xf64 alpha75 = BROADCAST_f64( *(a11 + 7 + 5 * 8), gvl ); // (alpha75)
	__epi_1xf64 alpha76 = BROADCAST_f64( *(a11 + 7 + 6 * 8), gvl ); // (alpha76)
	__epi_1xf64 alpha77 = BROADCAST_f64( *(a11 + 7 + 7 * 8), gvl ); // (1/alpha77)

	accum0 = __builtin_epi_vfmul_1xf64( cv00, alpha70, gvl );            // 
	accum1 = __builtin_epi_vfmul_1xf64( cv01, alpha70, gvl );            // 
	accum2 = __builtin_epi_vfmul_1xf64( cv02, alpha70, gvl );            // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha71, cv10, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha71, cv11, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha71, cv12, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha72, cv20, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha72, cv21, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha72, cv22, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha73, cv30, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha73, cv31, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha73, cv32, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha74, cv40, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha74, cv41, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha74, cv42, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha75, cv50, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha75, cv51, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha75, cv52, gvl );    // 

	accum0 = __builtin_epi_vfmacc_1xf64( accum0, alpha76, cv60, gvl );    // 
	accum1 = __builtin_epi_vfmacc_1xf64( accum1, alpha76, cv61, gvl );    // 
	accum2 = __builtin_epi_vfmacc_1xf64( accum2, alpha76, cv62, gvl );    // 

	cv70 = __builtin_epi_vfsub_1xf64( cv70, accum0, gvl );    // 
	cv71 = __builtin_epi_vfsub_1xf64( cv71, accum1, gvl );    // 
	cv72 = __builtin_epi_vfsub_1xf64( cv72, accum2, gvl );    // 

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	cv70 = __builtin_epi_vfmul_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
	cv71 = __builtin_epi_vfmul_1xf64( cv71, alpha77, gvl ); // abv0 *= alpha00
	cv72 = __builtin_epi_vfmul_1xf64( cv72, alpha77, gvl ); // abv0 *= alpha00
#else
	cv70 = __builtin_epi_vfdiv_1xf64( cv70, alpha77, gvl ); // abv0 *= alpha00
	cv71 = __builtin_epi_vfdiv_1xf64( cv71, alpha77, gvl ); // abv0 *= alpha00
	cv72 = __builtin_epi_vfdiv_1xf64( cv72, alpha77, gvl ); // abv0 *= alpha00
#endif

	__builtin_epi_vstore_1xf64( b11 + 7 * nr + 0 * vlen * cs_c, cv70, gvl ); // Store row 8 of b11
	__builtin_epi_vstore_1xf64( b11 + 7 * nr + 1 * vlen * cs_c, cv71, gvl ); // Store row 8 of b11
	__builtin_epi_vstore_1xf64( b11 + 7 * nr + 2 * vlen * cs_c, cv72, gvl ); // Store row 8 of b11

	//__builtin_epi_vstore_1xf64( b11 + 21*vlen + 0*cs_c+0*vlen, cv70, gvl ); // Store row 8 of b11
	//__builtin_epi_vstore_1xf64( b11 + 22*vlen + 0*cs_c+0*vlen, cv71, gvl ); // Store row 8 of b11
	//__builtin_epi_vstore_1xf64( b11 + 23*vlen + 0*cs_c+0*vlen, cv72, gvl ); // Store row 8 of b11

	// END TRSM

	// Row-major.
	if( cs_c == 1 )
	{
		//printf("rs_c = %d, nr = %d, vlen = %d\n", rs_c, nr, vlen);
		// Store row 0
		__builtin_epi_vstore_1xf64( c11 + 0*nr + cs_c*0*vlen,    cv00, gvl );
		__builtin_epi_vstore_1xf64( c11 + 0*nr + cs_c*1*vlen,    cv01, gvl );
		__builtin_epi_vstore_1xf64( c11 + 0*nr + cs_c*2*vlen,    cv02, gvl );
		// Store row 1
		__builtin_epi_vstore_1xf64( c11 + 1*nr + cs_c*0*vlen,    cv10, gvl );
		__builtin_epi_vstore_1xf64( c11 + 1*nr + cs_c*1*vlen,    cv11, gvl );
		__builtin_epi_vstore_1xf64( c11 + 1*nr + cs_c*2*vlen,    cv12, gvl );
		// Store row 2
		__builtin_epi_vstore_1xf64( c11 + 2*nr + cs_c*0*vlen,    cv20, gvl );
		__builtin_epi_vstore_1xf64( c11 + 2*nr + cs_c*1*vlen,    cv21, gvl );
		__builtin_epi_vstore_1xf64( c11 + 2*nr + cs_c*2*vlen,    cv22, gvl );
		// Store row 3
		__builtin_epi_vstore_1xf64( c11 + 3*nr + cs_c*0*vlen,    cv30, gvl );
		__builtin_epi_vstore_1xf64( c11 + 3*nr + cs_c*1*vlen,    cv31, gvl );
		__builtin_epi_vstore_1xf64( c11 + 3*nr + cs_c*2*vlen,    cv32, gvl );
		// Store row 4
		__builtin_epi_vstore_1xf64( c11 + 4*nr + cs_c*0*vlen,    cv40, gvl );
		__builtin_epi_vstore_1xf64( c11 + 4*nr + cs_c*1*vlen,    cv41, gvl );
		__builtin_epi_vstore_1xf64( c11 + 4*nr + cs_c*2*vlen,    cv42, gvl );
		// Store row 5
		__builtin_epi_vstore_1xf64( c11 + 5*nr + cs_c*0*vlen,    cv50, gvl );
		__builtin_epi_vstore_1xf64( c11 + 5*nr + cs_c*1*vlen,    cv51, gvl );
		__builtin_epi_vstore_1xf64( c11 + 5*nr + cs_c*2*vlen,    cv52, gvl );
		// Store row 6
		__builtin_epi_vstore_1xf64( c11 + 6*nr + cs_c*0*vlen,    cv60, gvl );
		__builtin_epi_vstore_1xf64( c11 + 6*nr + cs_c*1*vlen,    cv61, gvl );
		__builtin_epi_vstore_1xf64( c11 + 6*nr + cs_c*2*vlen,    cv62, gvl );
		// Store row 7
		__builtin_epi_vstore_1xf64( c11 + 7*nr + cs_c*0*vlen,    cv70, gvl );
		__builtin_epi_vstore_1xf64( c11 + 7*nr + cs_c*1*vlen,    cv71, gvl );
		__builtin_epi_vstore_1xf64( c11 + 7*nr + cs_c*2*vlen,    cv72, gvl );
	}
#if 1
	// Column-major.
	if( rs_c == 1 )
	{
		printf("This was rs_c == 1\n");
#if 0
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
#endif

	}
#endif

	GEMMTRSM_UKR_FLUSH_CT( d );
}
