/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

void bli_cntx_init_epi( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_epi_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_ukrs
	(
	  cntx, 

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,          bli_sgemm_epi_scalar_8x3v,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,         bli_dgemm_epi_scalar_8x3v,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_l_epi_scalar_8x3v,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_u_epi_scalar_8x3v,

	  // gemmsup
	  BLIS_GEMMSUP_RRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //d x8m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //d x8n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_epi_int_8x3vm, //x8n,

	  //BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16m,
	  //BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_epi_int_8x3vm, //x16m,
	  //BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16m,
	  //BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16n,
	  //BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16m,
	  //BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_epi_int_8x3vm, //x16n,
	  //BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16n,
	  //BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_epi_int_8x3vm, //x16n,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  // gemmtrsm_l
	  BLIS_GEMMTRSM_L_UKR_ROW_PREF, BLIS_FLOAT,    TRUE,
	  BLIS_GEMMTRSM_L_UKR_ROW_PREF, BLIS_DOUBLE,   TRUE,

	  // gemmtrsm_u
	  BLIS_GEMMTRSM_U_UKR_ROW_PREF, BLIS_FLOAT,    TRUE,
	  BLIS_GEMMTRSM_U_UKR_ROW_PREF, BLIS_DOUBLE,   TRUE,

	  // gemmsup
	  BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  //BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	  //BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,

	  BLIS_VA_END
	);


#if 0
	bli_cntx_set_l1v_kers
		(
		 12,
	  BLIS_AXPYV_KER,  BLIS_FLOAT,  bli_saxpyv_epi_int,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE, bli_daxpyv_epi_int,
	  BLIS_AMAXV_KER,  BLIS_FLOAT,  bli_samaxv_epi_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE, bli_damaxv_epi_int,
 	  BLIS_SCALV_KER,  BLIS_FLOAT,  bli_sscalv_epi_int,
	  BLIS_SCALV_KER,  BLIS_DOUBLE, bli_dscalv_epi_int,
 	  BLIS_SWAPV_KER,  BLIS_FLOAT,  bli_sswapv_epi_int,
	  BLIS_SWAPV_KER,  BLIS_DOUBLE, bli_dswapv_epi_int,
	  BLIS_COPYV_KER,  BLIS_FLOAT,  bli_scopyv_epi_int,
	  BLIS_COPYV_KER,  BLIS_DOUBLE, bli_dcopyv_epi_int,
          BLIS_SETV_KER,   BLIS_FLOAT,  bli_ssetv_epi_int,
          BLIS_SETV_KER,   BLIS_DOUBLE, bli_dsetv_epi_int,
		 cntx
		);
#endif

	unsigned long int vector_length_sp;
	unsigned long int vector_length_dp;

	//vector_length_sp = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
	vector_length_sp = 240;
	//vector_length_dp = __builtin_epi_vsetvlmax(__epi_e64, __epi_m1);
	vector_length_dp = 240;

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     8,                    8,                    0, 0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],     vector_length_sp * 3, vector_length_dp * 3, 0, 0 );
	//
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],  1024,   1024,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,   512,     0,     0 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NC ],  3600,  3600,     0,     0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  7200,  7200,     0,     0 );

        // Initialize sup thresholds with architecture-appropriate values.
	//                                          s     d     c     z
	bli_blksz_init_easy( &blkszs[ BLIS_MT ],  8001,  8001,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],  8001,  8001,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],  8001,  8001,   -1,   -1 );

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR_SUP ],     8,     8,    -1,    -1,
	                                                 8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],    vector_length_sp * 3,     vector_length_dp * 3,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],  1024,  1024,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],   512,   720,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  7200,  7200,    -1,    -1 );
	//bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],   168,    72,    -1,    -1 );
	//bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],   256,   256,    -1,    -1 );
	//bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  3600,  3600,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  cntx,

	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

	  // gemmsup thresholds
	  BLIS_MT, &blkszs[ BLIS_MT ], BLIS_MT,
	  BLIS_NT, &blkszs[ BLIS_NT ], BLIS_NT,
	  BLIS_KT, &blkszs[ BLIS_KT ], BLIS_KT,

	  // level-3 sup
	  BLIS_NC_SUP, &blkszs[ BLIS_NC_SUP ], BLIS_NR_SUP,
	  BLIS_KC_SUP, &blkszs[ BLIS_KC_SUP ], BLIS_KR_SUP,
	  BLIS_MC_SUP, &blkszs[ BLIS_MC_SUP ], BLIS_MR_SUP,
	  BLIS_NR_SUP, &blkszs[ BLIS_NR_SUP ], BLIS_NR_SUP,
	  BLIS_MR_SUP, &blkszs[ BLIS_MR_SUP ], BLIS_MR_SUP,

	  BLIS_VA_END
	);

}

