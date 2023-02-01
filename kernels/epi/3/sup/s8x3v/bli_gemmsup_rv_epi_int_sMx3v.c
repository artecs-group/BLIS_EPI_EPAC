#include "blis.h"
	
// Prototype reference microkernels.
GEMMSUP_KER_PROT( float,   s, gemmsup_r_epi_ref )

void bli_sgemmsup_rv_epi_int_8x3v
     (
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
#if 1
	//printf( "  - REF: m0: %d; n0: %d; k0: %d;\n", m0, n0, k0 );
	bli_sgemmsup_r_epi_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	//printf( "  - DONE REF: m0: %d; n0: %d; k0: %d;\n", m0, n0, k0 );
	return;
#endif
}
