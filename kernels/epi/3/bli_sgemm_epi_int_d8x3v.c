#include "blis.h"

#ifdef EPI_FPGA
#define BROADCAST_f32 __builtin_epi_vbroadcast_2xf32
#else
#define BROADCAST_f32 __builtin_epi_vfmv_v_f_2xf32
#endif

void bli_sgemm_epi_scalar_8x3v
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint32_t k_iter = k / 1;
	//uint32_t k_left = k % 1;
	uint32_t rs_c = rs_c0;
	uint32_t cs_c = cs_c0;
	uint32_t i;

	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

        const dim_t     nr     = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx ); 


	GEMM_UKR_SETUP_CT( s, bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_MR, cntx ),
        		      bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx ),
			      true ); 

	//printf( "Microkernel: m: %d, n: %d, k: %d\n", m, n, k );

	long gvl = __builtin_epi_vsetvl( nr/3, __epi_e32, __epi_m1 );

        //unsigned long int vlen = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);
        //unsigned long int vlen = 240;
        unsigned long int vlen = 480;

	// B vectors.
	__epi_2xf32 bv00, bv01, bv02;

	// C row (3v).
	__epi_2xf32 cv0, cv1, cv2;

	// Accummulators (8x3).
	__epi_2xf32 abv00, abv01, abv02, 
		    abv10, abv11, abv12, 
		    abv20, abv21, abv22, 
		    abv30, abv31, abv32, 
		    abv40, abv41, abv42, 
		    abv50, abv51, abv52, 
		    abv60, abv61, abv62, 
		    abv70, abv71, abv72;

	// Initialize accummulators to 0.0 (row 0)
	abv00 = BROADCAST_f32( 0.0, gvl );
	abv01 = BROADCAST_f32( 0.0, gvl );
	abv02 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 1)
	abv10 = BROADCAST_f32( 0.0, gvl );
	abv11 = BROADCAST_f32( 0.0, gvl );
	abv12 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 2)
	abv20 = BROADCAST_f32( 0.0, gvl );
	abv21 = BROADCAST_f32( 0.0, gvl );
	abv22 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 3)
	abv30 = BROADCAST_f32( 0.0, gvl );
	abv31 = BROADCAST_f32( 0.0, gvl );
	abv32 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 4)
	abv40 = BROADCAST_f32( 0.0, gvl );
	abv41 = BROADCAST_f32( 0.0, gvl );
	abv42 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 5)
	abv50 = BROADCAST_f32( 0.0, gvl );
	abv51 = BROADCAST_f32( 0.0, gvl );
	abv52 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 6)
	abv60 = BROADCAST_f32( 0.0, gvl );
	abv61 = BROADCAST_f32( 0.0, gvl );
	abv62 = BROADCAST_f32( 0.0, gvl );

	// Initialize accummulators to 0.0 (row 7)
	abv70 = BROADCAST_f32( 0.0, gvl );
	abv71 = BROADCAST_f32( 0.0, gvl );
	abv72 = BROADCAST_f32( 0.0, gvl );

	__epi_2xf32 sav1;

	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iteration 0
 		bv00 = __builtin_epi_vload_2xf32( b+0*vlen, gvl );
 		bv01 = __builtin_epi_vload_2xf32( b+1*vlen, gvl );
 		bv02 = __builtin_epi_vload_2xf32( b+2*vlen, gvl );

 		sav1 = BROADCAST_f32( *(a), gvl );
		abv00 = __builtin_epi_vfmacc_2xf32( abv00, sav1, bv00, gvl );
		abv01 = __builtin_epi_vfmacc_2xf32( abv01, sav1, bv01, gvl );
		abv02 = __builtin_epi_vfmacc_2xf32( abv02, sav1, bv02, gvl );

		sav1 = BROADCAST_f32( *(a+1), gvl );
		abv10 = __builtin_epi_vfmacc_2xf32( abv10, sav1, bv00, gvl );
		abv11 = __builtin_epi_vfmacc_2xf32( abv11, sav1, bv01, gvl );
		abv12 = __builtin_epi_vfmacc_2xf32( abv12, sav1, bv02, gvl );

 		sav1 = BROADCAST_f32( *(a+2), gvl );
		abv20 = __builtin_epi_vfmacc_2xf32( abv20, sav1, bv00, gvl );
		abv21 = __builtin_epi_vfmacc_2xf32( abv21, sav1, bv01, gvl );
		abv22 = __builtin_epi_vfmacc_2xf32( abv22, sav1, bv02, gvl );

		sav1 = BROADCAST_f32( *(a+3), gvl );
		abv30 = __builtin_epi_vfmacc_2xf32( abv30, sav1, bv00, gvl );
		abv31 = __builtin_epi_vfmacc_2xf32( abv31, sav1, bv01, gvl );
		abv32 = __builtin_epi_vfmacc_2xf32( abv32, sav1, bv02, gvl );

 		sav1 = BROADCAST_f32( *(a+4), gvl );
		abv40 = __builtin_epi_vfmacc_2xf32( abv40, sav1, bv00, gvl );
		abv41 = __builtin_epi_vfmacc_2xf32( abv41, sav1, bv01, gvl );
		abv42 = __builtin_epi_vfmacc_2xf32( abv42, sav1, bv02, gvl );

		sav1 = BROADCAST_f32( *(a+5), gvl );
		abv50 = __builtin_epi_vfmacc_2xf32( abv50, sav1, bv00, gvl );
		abv51 = __builtin_epi_vfmacc_2xf32( abv51, sav1, bv01, gvl );
		abv52 = __builtin_epi_vfmacc_2xf32( abv52, sav1, bv02, gvl );

 		sav1 = BROADCAST_f32( *(a+6), gvl );
		abv60 = __builtin_epi_vfmacc_2xf32( abv60, sav1, bv00, gvl );
		abv61 = __builtin_epi_vfmacc_2xf32( abv61, sav1, bv01, gvl );
		abv62 = __builtin_epi_vfmacc_2xf32( abv62, sav1, bv02, gvl );

		sav1 = BROADCAST_f32( *(a+7), gvl );
		abv70 = __builtin_epi_vfmacc_2xf32( abv70, sav1, bv00, gvl );
		abv71 = __builtin_epi_vfmacc_2xf32( abv71, sav1, bv01, gvl );
		abv72 = __builtin_epi_vfmacc_2xf32( abv72, sav1, bv02, gvl );

 		// Adjust pointers for next iterations.
	        b += 3 * vlen * 1; // 1 vectors + 1 Unroll factor
		a += 8;
	}

#if 0
	for ( i = 0; i < k_left; ++i )
	{
 		av00 = __builtin_epi_vload_2xf32( a+0*vlen, gvl );

 		sbv1 = BROADCAST_f32( *(b), gvl );
		abv00 = __builtin_epi_vfmacc_2xf32( abv00, sbv1, av00, gvl );

		sbv1 = BROADCAST_f32( *(b+1), gvl );
		abv01 = __builtin_epi_vfmacc_2xf32( abv01, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f32( *(b+2), gvl );
		abv02 = __builtin_epi_vfmacc_2xf32( abv02, sbv1, av00, gvl );

		sbv1 = BROADCAST_f32( *(b+3), gvl );
		abv03 = __builtin_epi_vfmacc_2xf32( abv03, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f32( *(b+4), gvl );
		abv04 = __builtin_epi_vfmacc_2xf32( abv04, sbv1, av00, gvl );

		sbv1 = BROADCAST_f32( *(b+5), gvl );
		abv05 = __builtin_epi_vfmacc_2xf32( abv05, sbv1, av00, gvl );

 		sbv1 = BROADCAST_f32( *(b+6), gvl );
		abv06 = __builtin_epi_vfmacc_2xf32( abv06, sbv1, av00, gvl );

		sbv1 = BROADCAST_f32( *(b+7), gvl );
		abv07 = __builtin_epi_vfmacc_2xf32( abv07, sbv1, av00, gvl );

		a += 1*vlen;
		b += 24;
	}
#endif


	// Scale by alpha
	__epi_2xf32 alphav;
	alphav = BROADCAST_f32( *alpha, gvl );

	abv00 = __builtin_epi_vfmul_2xf32( abv00, alphav, gvl );
	abv01 = __builtin_epi_vfmul_2xf32( abv01, alphav, gvl );
	abv02 = __builtin_epi_vfmul_2xf32( abv02, alphav, gvl );

	abv10 = __builtin_epi_vfmul_2xf32( abv10, alphav, gvl );
	abv11 = __builtin_epi_vfmul_2xf32( abv11, alphav, gvl );
	abv12 = __builtin_epi_vfmul_2xf32( abv12, alphav, gvl );

	abv20 = __builtin_epi_vfmul_2xf32( abv20, alphav, gvl );
	abv21 = __builtin_epi_vfmul_2xf32( abv21, alphav, gvl );
	abv22 = __builtin_epi_vfmul_2xf32( abv22, alphav, gvl );

	abv30 = __builtin_epi_vfmul_2xf32( abv30, alphav, gvl );
	abv31 = __builtin_epi_vfmul_2xf32( abv31, alphav, gvl );
	abv32 = __builtin_epi_vfmul_2xf32( abv32, alphav, gvl );

	abv40 = __builtin_epi_vfmul_2xf32( abv40, alphav, gvl );
	abv41 = __builtin_epi_vfmul_2xf32( abv41, alphav, gvl );
	abv42 = __builtin_epi_vfmul_2xf32( abv42, alphav, gvl );

	abv50 = __builtin_epi_vfmul_2xf32( abv50, alphav, gvl );
	abv51 = __builtin_epi_vfmul_2xf32( abv51, alphav, gvl );
	abv52 = __builtin_epi_vfmul_2xf32( abv52, alphav, gvl );

	abv60 = __builtin_epi_vfmul_2xf32( abv60, alphav, gvl );
	abv61 = __builtin_epi_vfmul_2xf32( abv61, alphav, gvl );
	abv62 = __builtin_epi_vfmul_2xf32( abv62, alphav, gvl );

	abv70 = __builtin_epi_vfmul_2xf32( abv70, alphav, gvl );
	abv71 = __builtin_epi_vfmul_2xf32( abv71, alphav, gvl );
	abv72 = __builtin_epi_vfmul_2xf32( abv72, alphav, gvl );

	if ( *beta != 0.0 ) {

	    __epi_2xf32 betav;
	    betav = BROADCAST_f32( *beta, gvl );


	    // Row-major.
	    if( cs_c == 1 )
	    {
		//printf("bnzero cs_c == 1; rs_c == %d\n", rs_c);
 		cv0 = __builtin_epi_vload_2xf32( c + 0*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv00, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 1*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv10, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 2*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv20, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 3*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv30, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 4*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv40, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 5*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv50, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 6*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv60, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 7*rs_c + 0*cs_c + 0*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv70, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, gvl );

		// ==== 
 		cv0 = __builtin_epi_vload_2xf32( c + 0*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv01, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 1*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv11, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 2*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv21, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 3*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv31, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 4*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv41, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 5*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv51, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 6*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv61, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 7*rs_c + 0*cs_c + 1*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv71, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+1*vlen*cs_c,    cv0, gvl );

		// ==
 		cv0 = __builtin_epi_vload_2xf32( c + 0*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv02, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 1*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv12, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 2*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv22, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 3*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv32, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 4*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv42, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 5*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv52, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 6*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv62, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );

 		cv0 = __builtin_epi_vload_2xf32( c + 7*rs_c + 0*cs_c + 2*vlen*cs_c, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( abv72, cv0, betav, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+2*vlen*cs_c,    cv0, gvl );


	    }

	    // Column major.
	    if( rs_c == 1 )
	    {
		printf("bnzero rs_c == 1; cs_c == %d\n", cs_c);
		// Load row 0
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv00, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv01, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv02, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 1
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv10, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv11, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv12, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 2
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv20, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv21, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv22, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 3
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv30, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv31, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv32, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 4
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv40, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv41, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv42, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 5
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv50, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv51, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv52, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 6
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv60, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv61, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv62, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 7
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmul_2xf32( cv0, betav, gvl );
	        cv1 = __builtin_epi_vfmul_2xf32( cv1, betav, gvl );
	        cv2 = __builtin_epi_vfmul_2xf32( cv2, betav, gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv70, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv71, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv72, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );
	    }

	}
	else { // Beta == 0

	// Row-major.
	if( cs_c == 1 )
	{
		//printf("bzero cs_c == 1; rs_c == %d\n", rs_c);
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+0*vlen*cs_c,    abv00, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+0*vlen*cs_c,    abv10, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+0*vlen*cs_c,    abv20, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+0*vlen*cs_c,    abv30, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+0*vlen*cs_c,    abv40, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+0*vlen*cs_c,    abv50, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+0*vlen*cs_c,    abv60, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+0*vlen*cs_c,    abv70, gvl );

		// ==== 
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+1*vlen*cs_c,    abv01, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+1*vlen*cs_c,    abv11, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+1*vlen*cs_c,    abv21, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+1*vlen*cs_c,    abv31, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+1*vlen*cs_c,    abv41, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+1*vlen*cs_c,    abv51, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+1*vlen*cs_c,    abv61, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+1*vlen*cs_c,    abv71, gvl );

		// ==
		__builtin_epi_vstore_2xf32( c + 0*rs_c + 0*cs_c+2*vlen*cs_c,    abv02, gvl );
		__builtin_epi_vstore_2xf32( c + 1*rs_c + 0*cs_c+2*vlen*cs_c,    abv12, gvl );
		__builtin_epi_vstore_2xf32( c + 2*rs_c + 0*cs_c+2*vlen*cs_c,    abv22, gvl );
		__builtin_epi_vstore_2xf32( c + 3*rs_c + 0*cs_c+2*vlen*cs_c,    abv32, gvl );
		__builtin_epi_vstore_2xf32( c + 4*rs_c + 0*cs_c+2*vlen*cs_c,    abv42, gvl );
		__builtin_epi_vstore_2xf32( c + 5*rs_c + 0*cs_c+2*vlen*cs_c,    abv52, gvl );
		__builtin_epi_vstore_2xf32( c + 6*rs_c + 0*cs_c+2*vlen*cs_c,    abv62, gvl );
		__builtin_epi_vstore_2xf32( c + 7*rs_c + 0*cs_c+2*vlen*cs_c,    abv72, gvl );

	}
	// Column major.
	if( rs_c == 1 )
	{
		printf("bzero rs_c == 1; cs_c == %d\n", cs_c);
		// Load row 0
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 0*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv00, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv01, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv02, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 0*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 1
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 1*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv10, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv11, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv12, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 1*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 2
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 2*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv20, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv21, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv22, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 2*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 3
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 3*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv30, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv31, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv32, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 3*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 4
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 4*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv40, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv41, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv42, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 4*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 5
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 5*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv50, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv51, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv52, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 5*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 6
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 6*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv60, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv61, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv62, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 6*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );

		// Load row 7
 		cv0 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 0*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv1 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 1*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
 		cv2 = __builtin_epi_vload_strided_2xf32( c + 7*rs_c + 0*cs_c + 2*vlen*cs_c, (long int)cs_c*sizeof(float) , gvl );
	        cv0 = __builtin_epi_vfmacc_2xf32( cv0, alphav, abv70, gvl );
	        cv1 = __builtin_epi_vfmacc_2xf32( cv1, alphav, abv71, gvl );
	        cv2 = __builtin_epi_vfmacc_2xf32( cv2, alphav, abv72, gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+0*vlen*cs_c,    cv0, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+1*vlen*cs_c,    cv1, (long int)cs_c*sizeof(float) , gvl );
		__builtin_epi_vstore_strided_2xf32( c + 7*rs_c + 0*cs_c+2*vlen*cs_c,    cv2, (long int)cs_c*sizeof(float) , gvl );
	}

	}

	GEMM_UKR_FLUSH_CT( s );

}

