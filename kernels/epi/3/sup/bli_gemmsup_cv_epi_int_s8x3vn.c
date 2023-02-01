#include "blis.h"

#ifdef EPI_FPGA
#define BROADCAST_f64 __builtin_epi_vbroadcast_2xf32
#else
#define BROADCAST_f64 __builtin_epi_vfmv_v_f_2xf32
#endif

// Prototype reference microkernels (TODO EPI).
//GEMMSUP_KER_PROT( double,   d, gemmsup_r_epi_ref )

void bli_sgemmsup_rv_epi_int_8x3vn (
		conj_t              conja,
		conj_t              conjb,
		dim_t               m0,
		dim_t               n0,
		dim_t               k0,
		float*    restrict alpha,
		float*    restrict a, inc_t rs_a0, inc_t cs_a0,
		float*    restrict b, inc_t rs_b0, inc_t cs_b0,
		float*    restrict beta,
		float*    restrict c, inc_t rs_c0, inc_t cs_c0,
		auxinfo_t*          data,
		cntx_t*             cntx
		)
{

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint32_t k_iter = k0 / 1;
	uint32_t k_left = k0 % 1;

	uint32_t n_iter = n0 / ( 3 * 480 ); //720; // 3 * vlen
	uint32_t n_left = n0 % ( 3 * 480 ); //720; // 3 * vlen

	uint32_t rs_a = rs_a0;
	uint32_t cs_a = cs_a0;
	uint32_t rs_b = rs_b0;
	uint32_t cs_b = cs_b0;
	uint32_t rs_c = rs_c0;
	uint32_t cs_c = cs_c0;

	uint32_t i, ii;

	float * a_ii, * b_ii, * c_ii;

	// Backup copies of a, b, c (for edge cases).
	float * a_bak = a;
	float * b_bak = b;
	float * c_bak = c;

	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

	// Query the panel stride of A and convert it to units of bytes.
	uint32_t ps_b   = bli_auxinfo_ps_b( data );
	uint32_t ps_b8  = ps_b * sizeof( float );

	printf( "Invoking bli_sgemmsup_rv_haswell_asm_6x8n( m0: %d, n0: %d, k0: %d,  rs_a0: %d, cs_a0: %d, rs_b0: %d, cs_b0: %d, rs_c0: %d, cs_c0: %d, ps_b: %d, alpha: %f, beta: %f\n", m0, n0, k0, rs_a0, cs_a0, rs_b0, cs_b0, rs_c0, cs_c0, ps_b, *alpha, *beta );

	const dim_t     nr     = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_NR, cntx ); 

	//unsigned long int vlen = 240;
	unsigned long int vlen = 480;

	// Calculate effective usage of 3v vectors.
	int effu_1 = ( n0 >= vlen ) ? vlen : n0; effu_1 = ( effu_1 < 0 ) ? 0 : effu_1;
	int effu_2 = ( n0 >= 2 * vlen ) ? vlen : n0 - vlen; effu_2 = ( effu_2 < 0 ) ? 0 : effu_2;
	int effu_3 = ( n0 >= 3 * vlen ) ? vlen : n0 - 2 * vlen; effu_3 = ( effu_3 < 0 ) ? 0 : effu_3;

	long gvl1 = __builtin_epi_vsetvl( effu_1, __epi_e32, __epi_m1 );
	long gvl2 = __builtin_epi_vsetvl( effu_2, __epi_e32, __epi_m1 );
	long gvl3 = __builtin_epi_vsetvl( effu_3, __epi_e32, __epi_m1 );

	long gvl = __builtin_epi_vsetvl( nr/3, __epi_e32, __epi_m1 );

	printf( "m0: %d; n0: %d; k0: %d; effu_1: %d; effu_2: %d; effu_3: %d; nr: %d; n_iter: %d, n_left: %d\n", m0, n0, k0, effu_1, effu_2, effu_3, nr, n_iter, n_left );
	printf( "gvl: %ld; gvl1: %ld; gvl2: %ld; gvl3: %ld\n", gvl, gvl1, gvl2, gvl3 );

	if ( n_iter == 0 ) goto consider_edge_cases;

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

	__epi_2xf32 sav1;

	a_ii = a;
	b_ii = b;
	c_ii = c;

	for ( ii = 0; ii < n_iter; ++ii )
	{

		//printf(" --> Start ii: %d\n", ii);

		bv00 = BROADCAST_f64( 0.0, gvl1 );
		bv01 = BROADCAST_f64( 0.0, gvl2 );
		bv02 = BROADCAST_f64( 0.0, gvl3 );

		cv0 = BROADCAST_f64( 0.0, gvl1 );
		cv1 = BROADCAST_f64( 0.0, gvl2 );
		cv2 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 0)
		abv00 = BROADCAST_f64( 0.0, gvl1 );
		abv01 = BROADCAST_f64( 0.0, gvl2 );
		abv02 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 1)
		abv10 = BROADCAST_f64( 0.0, gvl1 );
		abv11 = BROADCAST_f64( 0.0, gvl2 );
		abv12 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 2)
		abv20 = BROADCAST_f64( 0.0, gvl1 );
		abv21 = BROADCAST_f64( 0.0, gvl2 );
		abv22 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 3)
		abv30 = BROADCAST_f64( 0.0, gvl1 );
		abv31 = BROADCAST_f64( 0.0, gvl2 );
		abv32 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 4)
		abv40 = BROADCAST_f64( 0.0, gvl1 );
		abv41 = BROADCAST_f64( 0.0, gvl2 );
		abv42 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 5)
		abv50 = BROADCAST_f64( 0.0, gvl1 );
		abv51 = BROADCAST_f64( 0.0, gvl2 );
		abv52 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 6)
		abv60 = BROADCAST_f64( 0.0, gvl1 );
		abv61 = BROADCAST_f64( 0.0, gvl2 );
		abv62 = BROADCAST_f64( 0.0, gvl3 );

		// Initialize accummulators to 0.0 (row 7)
		abv70 = BROADCAST_f64( 0.0, gvl1 );
		abv71 = BROADCAST_f64( 0.0, gvl2 );
		abv72 = BROADCAST_f64( 0.0, gvl3 );

		// Reload a, b, c on each m iteration
		a_ii = a;
		b_ii = b;
		c_ii = c;

		for ( i = 0; i < k_iter; ++i )
		{
			// Begin iteration 0
			bv00 = __builtin_epi_vload_2xf32( b_ii+0*vlen, gvl1 );
			bv01 = __builtin_epi_vload_2xf32( b_ii+1*vlen, gvl2 );
			bv02 = __builtin_epi_vload_2xf32( b_ii+2*vlen, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+0*rs_a), gvl );
			abv00 = __builtin_epi_vfmacc_2xf32( abv00, sav1, bv00, gvl1 );
			abv01 = __builtin_epi_vfmacc_2xf32( abv01, sav1, bv01, gvl2 );
			abv02 = __builtin_epi_vfmacc_2xf32( abv02, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+1*rs_a), gvl );
			abv10 = __builtin_epi_vfmacc_2xf32( abv10, sav1, bv00, gvl1 );
			abv11 = __builtin_epi_vfmacc_2xf32( abv11, sav1, bv01, gvl2 );
			abv12 = __builtin_epi_vfmacc_2xf32( abv12, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+2*rs_a), gvl );
			abv20 = __builtin_epi_vfmacc_2xf32( abv20, sav1, bv00, gvl1 );
			abv21 = __builtin_epi_vfmacc_2xf32( abv21, sav1, bv01, gvl2 );
			abv22 = __builtin_epi_vfmacc_2xf32( abv22, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+3*rs_a), gvl );
			abv30 = __builtin_epi_vfmacc_2xf32( abv30, sav1, bv00, gvl1 );
			abv31 = __builtin_epi_vfmacc_2xf32( abv31, sav1, bv01, gvl2 );
			abv32 = __builtin_epi_vfmacc_2xf32( abv32, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+4*rs_a), gvl );
			abv40 = __builtin_epi_vfmacc_2xf32( abv40, sav1, bv00, gvl1 );
			abv41 = __builtin_epi_vfmacc_2xf32( abv41, sav1, bv01, gvl2 );
			abv42 = __builtin_epi_vfmacc_2xf32( abv42, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+5*rs_a), gvl );
			abv50 = __builtin_epi_vfmacc_2xf32( abv50, sav1, bv00, gvl1 );
			abv51 = __builtin_epi_vfmacc_2xf32( abv51, sav1, bv01, gvl2 );
			abv52 = __builtin_epi_vfmacc_2xf32( abv52, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+6*rs_a), gvl );
			abv60 = __builtin_epi_vfmacc_2xf32( abv60, sav1, bv00, gvl1 );
			abv61 = __builtin_epi_vfmacc_2xf32( abv61, sav1, bv01, gvl2 );
			abv62 = __builtin_epi_vfmacc_2xf32( abv62, sav1, bv02, gvl3 );

			sav1 = BROADCAST_f64( *(a_ii+7*rs_a), gvl );
			abv70 = __builtin_epi_vfmacc_2xf32( abv70, sav1, bv00, gvl1 );
			abv71 = __builtin_epi_vfmacc_2xf32( abv71, sav1, bv01, gvl2 );
			abv72 = __builtin_epi_vfmacc_2xf32( abv72, sav1, bv02, gvl3 );

			// Adjust pointers for next iterations.
			b_ii += rs_b; 
			a_ii += cs_a;
		}


		// Scale by alpha
		__epi_2xf32 alphav;
		alphav = BROADCAST_f64( *alpha, gvl );

		abv00 = __builtin_epi_vfmul_2xf32( abv00, alphav, gvl1 );
		abv01 = __builtin_epi_vfmul_2xf32( abv01, alphav, gvl2 );
		abv02 = __builtin_epi_vfmul_2xf32( abv02, alphav, gvl3 );

		abv10 = __builtin_epi_vfmul_2xf32( abv10, alphav, gvl1 );
		abv11 = __builtin_epi_vfmul_2xf32( abv11, alphav, gvl2 );
		abv12 = __builtin_epi_vfmul_2xf32( abv12, alphav, gvl3 );

		abv20 = __builtin_epi_vfmul_2xf32( abv20, alphav, gvl1 );
		abv21 = __builtin_epi_vfmul_2xf32( abv21, alphav, gvl2 );
		abv22 = __builtin_epi_vfmul_2xf32( abv22, alphav, gvl3 );

		abv30 = __builtin_epi_vfmul_2xf32( abv30, alphav, gvl1 );
		abv31 = __builtin_epi_vfmul_2xf32( abv31, alphav, gvl2 );
		abv32 = __builtin_epi_vfmul_2xf32( abv32, alphav, gvl3 );

		abv40 = __builtin_epi_vfmul_2xf32( abv40, alphav, gvl1 );
		abv41 = __builtin_epi_vfmul_2xf32( abv41, alphav, gvl2 );
		abv42 = __builtin_epi_vfmul_2xf32( abv42, alphav, gvl3 );

		abv50 = __builtin_epi_vfmul_2xf32( abv50, alphav, gvl1 );
		abv51 = __builtin_epi_vfmul_2xf32( abv51, alphav, gvl2 );
		abv52 = __builtin_epi_vfmul_2xf32( abv52, alphav, gvl3 );

		abv60 = __builtin_epi_vfmul_2xf32( abv60, alphav, gvl1 );
		abv61 = __builtin_epi_vfmul_2xf32( abv61, alphav, gvl2 );
		abv62 = __builtin_epi_vfmul_2xf32( abv62, alphav, gvl3 );

		abv70 = __builtin_epi_vfmul_2xf32( abv70, alphav, gvl1 );
		abv71 = __builtin_epi_vfmul_2xf32( abv71, alphav, gvl2 );
		abv72 = __builtin_epi_vfmul_2xf32( abv72, alphav, gvl3 );

		if ( *beta != 0.0 ) {

			__epi_2xf32 betav;
			betav = BROADCAST_f64( *beta, gvl );


			// Row-major.
			if( cs_c == 1 )
			{
				//printf("bnzero rs_c == %d; cs_c == 1\n", rs_c);
				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv00, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv01, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv02, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv10, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv11, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv12, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv20, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv21, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv22, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv30, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv31, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv32, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv40, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv41, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv42, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv50, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv51, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv52, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv60, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv61, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv62, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

				cv0 = __builtin_epi_vload_2xf32( c_ii + 0*vlen, gvl1 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv70, cv0, betav, gvl1 );
				__builtin_epi_vstore_2xf32( c_ii + 0*vlen, cv0, gvl1 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 1*vlen, gvl2 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv71, cv0, betav, gvl2 );
				__builtin_epi_vstore_2xf32( c_ii + 1*vlen, cv0, gvl2 );

				cv0 = __builtin_epi_vload_2xf32( c_ii + 2*vlen, gvl3 );
				cv0 = __builtin_epi_vfmacc_2xf32( abv72, cv0, betav, gvl3 );
				__builtin_epi_vstore_2xf32( c_ii + 2*vlen, cv0, gvl3 );

				c_ii += rs_c;

			}

			// Column major. (TODO EPI gvl)
			if( rs_c == 1 )
			{
				//printf("bnzero rs_c == 1; cs_c == %d\n", cs_c);
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
				//printf("bzero rs_c == 1; cs_c == %d\n", cs_c);
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

		//a = a + ps_a;
		b = b + ps_b;
		//c_ii = c_ii + 8 * rs_c;
		//c = c + 8*rs_c;
		c = c + 3*vlen*cs_c;

		//printf(" --> End ii: %d\n", ii);

	}

consider_edge_cases:;

		    if( n_left ) {
		            //printf("Considering edge cases\n");
			    const dim_t      mr_cur = 8;
			    const dim_t      j_edge = n0 - ( dim_t )n_left;

			    // Reference addresses of a, b, c are the backup ones.
			    //float* restrict cij = c_bak + i_edge*rs_c;
			    //float* restrict ai  = a_bak + m_iter * ps_a;
			    //float* restrict bj  = b_bak;

			    float* restrict cij = c_bak + j_edge * cs_c;
			    float* restrict ai  = a_bak;
			    float* restrict bj  = b_bak + n_iter * ps_b;

			    bli_sgemmsup_rv_epi_int_8x3v (
					    //conja, conjb, m_left, n0, k0,
					    conja, conjb, m0, n_left, k0,
					    alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					    beta, cij, rs_c0, cs_c0, data, cntx
					    );	
		            //printf("Done considering edge cases\n");
		    }

		    //GEMM_UKR_FLUSH_CT( d );
		    //printf("End microkernel call\n");

}

