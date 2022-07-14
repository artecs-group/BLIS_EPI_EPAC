#include "blis.h"

void bli_sswapv_epi_int
     (
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{

        const uint32_t     n_elem_per_reg = 8;
        uint32_t           i = 0;

        float* restrict x0;
        float* restrict y0;

        __epi_2xf32          xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7;
        __epi_2xf32          yv0, yv1, yv2, yv3, yv4, yv5, yv6, yv7;

	long gvl = __builtin_epi_vsetvl( 8, __epi_e32, __epi_m1 );
        
	// If the vector dimension is zero, return early.
        if ( bli_zero_dim1( n ) ) return;

        x0 = x;
        y0 = y;

        if ( incx == 1 && incy == 1 )
        {
                for ( i = 0; ( i + 63 ) < n; i += 64 )
                {
                        xv0 = __builtin_epi_vload_2xf32( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_2xf32( x0 + 1*n_elem_per_reg, gvl );
                        xv2 = __builtin_epi_vload_2xf32( x0 + 2*n_elem_per_reg, gvl );
                        xv3 = __builtin_epi_vload_2xf32( x0 + 3*n_elem_per_reg, gvl );
                        xv4 = __builtin_epi_vload_2xf32( x0 + 4*n_elem_per_reg, gvl );
                        xv5 = __builtin_epi_vload_2xf32( x0 + 5*n_elem_per_reg, gvl );
                        xv6 = __builtin_epi_vload_2xf32( x0 + 6*n_elem_per_reg, gvl );
                        xv7 = __builtin_epi_vload_2xf32( x0 + 7*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_2xf32( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_2xf32( y0 + 1*n_elem_per_reg, gvl );
                        yv2 = __builtin_epi_vload_2xf32( y0 + 2*n_elem_per_reg, gvl );
                        yv3 = __builtin_epi_vload_2xf32( y0 + 3*n_elem_per_reg, gvl );
                        yv4 = __builtin_epi_vload_2xf32( y0 + 4*n_elem_per_reg, gvl );
                        yv5 = __builtin_epi_vload_2xf32( y0 + 5*n_elem_per_reg, gvl );
                        yv6 = __builtin_epi_vload_2xf32( y0 + 6*n_elem_per_reg, gvl );
                        yv7 = __builtin_epi_vload_2xf32( y0 + 7*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_2xf32( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 1*n_elem_per_reg), yv1, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 2*n_elem_per_reg), yv2, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 3*n_elem_per_reg), yv3, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 4*n_elem_per_reg), yv4, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 5*n_elem_per_reg), yv5, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 6*n_elem_per_reg), yv6, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 7*n_elem_per_reg), yv7, gvl);

                        __builtin_epi_vstore_2xf32( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 1*n_elem_per_reg), xv1, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 2*n_elem_per_reg), xv2, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 3*n_elem_per_reg), xv3, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 4*n_elem_per_reg), xv4, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 5*n_elem_per_reg), xv5, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 6*n_elem_per_reg), xv6, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 7*n_elem_per_reg), xv7, gvl);

                        x0 += 8*n_elem_per_reg;
                        y0 += 8*n_elem_per_reg;
                }

                for ( ; ( i + 31 ) < n; i += 32 )
                {
                        xv0 = __builtin_epi_vload_2xf32( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_2xf32( x0 + 1*n_elem_per_reg, gvl );
                        xv2 = __builtin_epi_vload_2xf32( x0 + 2*n_elem_per_reg, gvl );
                        xv3 = __builtin_epi_vload_2xf32( x0 + 3*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_2xf32( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_2xf32( y0 + 1*n_elem_per_reg, gvl );
                        yv2 = __builtin_epi_vload_2xf32( y0 + 2*n_elem_per_reg, gvl );
                        yv3 = __builtin_epi_vload_2xf32( y0 + 3*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_2xf32( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 1*n_elem_per_reg), xv1, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 2*n_elem_per_reg), xv2, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 3*n_elem_per_reg), xv3, gvl);

                        __builtin_epi_vstore_2xf32( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 1*n_elem_per_reg), yv1, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 2*n_elem_per_reg), yv2, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 3*n_elem_per_reg), yv3, gvl);

                        x0 += 4*n_elem_per_reg;
                        y0 += 4*n_elem_per_reg;
                }

                for ( ; ( i + 15 ) < n; i += 16 )
                {
                        xv0 = __builtin_epi_vload_2xf32( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_2xf32( x0 + 1*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_2xf32( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_2xf32( y0 + 1*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_2xf32( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_2xf32( (y0 + 1*n_elem_per_reg), xv1, gvl);

                        __builtin_epi_vstore_2xf32( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_2xf32( (x0 + 1*n_elem_per_reg), yv1, gvl);

                        x0 += 2*n_elem_per_reg;
                        y0 += 2*n_elem_per_reg;
                }

                for ( ; ( i + 7 ) < n; i += 8 )
                {
                        xv0 = __builtin_epi_vload_2xf32( x0 + 0*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_2xf32( y0 + 0*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_2xf32( (x0 + 0*n_elem_per_reg), yv0, gvl);

                        __builtin_epi_vstore_2xf32( (y0 + 0*n_elem_per_reg), xv0, gvl);

                        x0 += 1*n_elem_per_reg;
                        y0 += 1*n_elem_per_reg;
                }

                for ( ; (i + 0) < n; i += 1 )
                {
                        PASTEMAC(s,swaps)( x[i], y[i] );
                }
        }
        else
        {
                for ( i = 0; i < n; ++i )
                {
                        PASTEMAC(s,swaps)( (*x0), (*y0) );

                        x0 += incx;
                        y0 += incy;
                }
        }

}

//--------------------------------------------------------------------------------


void bli_dswapv_epi_int
     (
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
        const uint32_t      n_elem_per_reg = 4;
        uint32_t            i = 0;

        double* restrict x0;
        double* restrict y0;

        __epi_1xf64        xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7;
        __epi_1xf64        yv0, yv1, yv2, yv3, yv4, yv5, yv6, yv7;
	
	long gvl = __builtin_epi_vsetvl( 4, __epi_e64, __epi_m1 );

        // If the vector dimension is zero, return early.
        if ( bli_zero_dim1( n ) ) return;

        x0 = x;
        y0 = y;

        if ( incx == 1 && incy == 1 )
        {
                for ( ; ( i + 31 ) < n; i += 32 )
                {
                        xv0 = __builtin_epi_vload_1xf64( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_1xf64( x0 + 1*n_elem_per_reg, gvl );
                        xv2 = __builtin_epi_vload_1xf64( x0 + 2*n_elem_per_reg, gvl );
                        xv3 = __builtin_epi_vload_1xf64( x0 + 3*n_elem_per_reg, gvl );
                        xv4 = __builtin_epi_vload_1xf64( x0 + 4*n_elem_per_reg, gvl );
                        xv5 = __builtin_epi_vload_1xf64( x0 + 5*n_elem_per_reg, gvl );
                        xv6 = __builtin_epi_vload_1xf64( x0 + 6*n_elem_per_reg, gvl );
                        xv7 = __builtin_epi_vload_1xf64( x0 + 7*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_1xf64( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_1xf64( y0 + 1*n_elem_per_reg, gvl );
                        yv2 = __builtin_epi_vload_1xf64( y0 + 2*n_elem_per_reg, gvl );
                        yv3 = __builtin_epi_vload_1xf64( y0 + 3*n_elem_per_reg, gvl );
                        yv4 = __builtin_epi_vload_1xf64( y0 + 4*n_elem_per_reg, gvl );
                        yv5 = __builtin_epi_vload_1xf64( y0 + 5*n_elem_per_reg, gvl );
                        yv6 = __builtin_epi_vload_1xf64( y0 + 6*n_elem_per_reg, gvl );
                        yv7 = __builtin_epi_vload_1xf64( y0 + 7*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_1xf64( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 1*n_elem_per_reg), yv1, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 2*n_elem_per_reg), yv2, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 3*n_elem_per_reg), yv3, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 4*n_elem_per_reg), yv4, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 5*n_elem_per_reg), yv5, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 6*n_elem_per_reg), yv6, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 7*n_elem_per_reg), yv7, gvl);

                        __builtin_epi_vstore_1xf64( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 1*n_elem_per_reg), xv1, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 2*n_elem_per_reg), xv2, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 3*n_elem_per_reg), xv3, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 4*n_elem_per_reg), xv4, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 5*n_elem_per_reg), xv5, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 6*n_elem_per_reg), xv6, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 7*n_elem_per_reg), xv7, gvl);

                        x0 += 8*n_elem_per_reg;
                        y0 += 8*n_elem_per_reg;
                }

                for ( ; ( i + 15 ) < n; i += 16 )
                {
                        xv0 = __builtin_epi_vload_1xf64( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_1xf64( x0 + 1*n_elem_per_reg, gvl );
                        xv2 = __builtin_epi_vload_1xf64( x0 + 2*n_elem_per_reg, gvl );
                        xv3 = __builtin_epi_vload_1xf64( x0 + 3*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_1xf64( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_1xf64( y0 + 1*n_elem_per_reg, gvl );
                        yv2 = __builtin_epi_vload_1xf64( y0 + 2*n_elem_per_reg, gvl );
                        yv3 = __builtin_epi_vload_1xf64( y0 + 3*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_1xf64( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 1*n_elem_per_reg), xv1, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 2*n_elem_per_reg), xv2, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 3*n_elem_per_reg), xv3, gvl);

                        __builtin_epi_vstore_1xf64( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 1*n_elem_per_reg), yv1, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 2*n_elem_per_reg), yv2, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 3*n_elem_per_reg), yv3, gvl);

                        x0 += 4*n_elem_per_reg;
                        y0 += 4*n_elem_per_reg;
                }

                for ( ; ( i + 7 ) < n; i += 8 )
                {
                        xv0 = __builtin_epi_vload_1xf64( x0 + 0*n_elem_per_reg, gvl );
                        xv1 = __builtin_epi_vload_1xf64( x0 + 1*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_1xf64( y0 + 0*n_elem_per_reg, gvl );
                        yv1 = __builtin_epi_vload_1xf64( y0 + 1*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_1xf64( (y0 + 0*n_elem_per_reg), xv0, gvl);
                        __builtin_epi_vstore_1xf64( (y0 + 1*n_elem_per_reg), xv1, gvl);

                        __builtin_epi_vstore_1xf64( (x0 + 0*n_elem_per_reg), yv0, gvl);
                        __builtin_epi_vstore_1xf64( (x0 + 1*n_elem_per_reg), yv1, gvl);

                        x0 += 2*n_elem_per_reg;
                        y0 += 2*n_elem_per_reg;
                }

                for ( ; ( i + 3 ) < n; i += 4 )
                {
                        xv0 = __builtin_epi_vload_1xf64( x0 + 0*n_elem_per_reg, gvl );

                        yv0 = __builtin_epi_vload_1xf64( y0 + 0*n_elem_per_reg, gvl );

                        __builtin_epi_vstore_1xf64( (y0 + 0*n_elem_per_reg), xv0, gvl);

                        __builtin_epi_vstore_1xf64( (x0 + 0*n_elem_per_reg), yv0, gvl);

                        x0 += 1*n_elem_per_reg;
                        y0 += 1*n_elem_per_reg;
                }

                for ( ; (i + 0) < n; i += 1 )
                {
                        PASTEMAC(d,swaps)( x[i], y[i] );
                }
        }
        else
        {
                for ( i = 0; i < n; ++i )
                {
                        PASTEMAC(d,swaps)( (*x0), (*y0) );

                        x0 += incx;
                        y0 += incy;
                }
        }
	
}

