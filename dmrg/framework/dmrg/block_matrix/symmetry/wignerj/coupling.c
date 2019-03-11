/* specfunc/coupling.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2002 Gerard Jungman
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* Author:  G. Jungman */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GSL_MAX(a,b) ((a) > (b) ? (a) : (b))
#define GSL_MAX_DBL(a,b)   GSL_MAX(a,b)
#define GSL_MIN(a,b) ((a) < (b) ? (a) : (b))
#define GSL_IS_ODD(n)  ((n) & 1)
#define GSL_IS_EVEN(n) (!(GSL_IS_ODD(n)))

#define GSL_SUCCESS 0
#define GSL_DBL_EPSILON    2.2204460492503131e-16
#define GSL_LOG_DBL_MAX    7.0978271289338397e+02
#define GSL_LOG_DBL_MIN   (-7.0839641853226408e+02)

#define DOMAIN_ERROR(result) printf("domain_error\n"); exit(1);
#define OVERFLOW_ERROR(result) printf("overflow_error\n"); exit(1);
#define UNDERFLOW_ERROR(result) printf("underflow_error\n"); exit(1);


inline
static
int locMax3(const int a, const int b, const int c)
{
  int d = GSL_MAX(a, b);
  return GSL_MAX(d, c);
}

inline
static
int locMin3(const int a, const int b, const int c)
{
  int d = GSL_MIN(a, b);
  return GSL_MIN(d, c);
}

inline
static
int locMin5(const int a, const int b, const int c, const int d, const int e)
{
  int f = GSL_MIN(a, b);
  int g = GSL_MIN(c, d);
  int h = GSL_MIN(f, g);
  return GSL_MIN(e, h);
}

struct gsl_sf_result_struct {
  double val;
  double err;
};
typedef struct gsl_sf_result_struct gsl_sf_result;


///////////////////////////////////////

#define LogRootTwoPi_  0.9189385332046727418

#define GSL_SF_FACT_NMAX 170
#include "fact_table.h"

int gsl_sf_fact_e(const unsigned int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n < 18) {
    result->val = fact_table[n].f;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else if(n <= GSL_SF_FACT_NMAX){
    result->val = fact_table[n].f;
    result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    OVERFLOW_ERROR(result);
  }
}

int gsl_sf_lnfact_e(const unsigned int n, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(n <= GSL_SF_FACT_NMAX){
    result->val = log(fact_table[n].f);
    result->err = 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
  else {
    printf("lngamma out of range\n");
    exit(1);
    //gsl_sf_lngamma_e(n+1.0, result);
    return GSL_SUCCESS;
  }
}


int gsl_sf_lnchoose_e(unsigned int n, unsigned int m, gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(m > n) {
    DOMAIN_ERROR(result);
  }
  else if(m == n || m == 0) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    gsl_sf_result nf;
    gsl_sf_result mf;
    gsl_sf_result nmmf;
    if(m*2 > n) m = n-m;
    gsl_sf_lnfact_e(n, &nf);
    gsl_sf_lnfact_e(m, &mf);
    gsl_sf_lnfact_e(n-m, &nmmf);
    result->val  = nf.val - mf.val - nmmf.val;
    result->err  = nf.err + mf.err + nmmf.err;
    result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
}

int
gsl_sf_exp_err_e(const double x, const double dx, gsl_sf_result * result)
{
  const double adx = fabs(dx);

  /* CHECK_POINTER(result) */

  if(x + adx > GSL_LOG_DBL_MAX) {
    OVERFLOW_ERROR(result);
  }
  else if(x - adx < GSL_LOG_DBL_MIN) {
    UNDERFLOW_ERROR(result);
  }
  else {
    const double ex  = exp(x);
    const double edx = exp(adx);
    result->val  = ex;
    result->err  = ex * GSL_MAX_DBL(GSL_DBL_EPSILON, edx - 1.0/edx);
    result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);
    return GSL_SUCCESS;
  }
}



///////////////////////////////////////


/* See: [Thompson, Atlas for Computing Mathematical Functions] */

static
int
delta(int ta, int tb, int tc, gsl_sf_result * d)
{
  gsl_sf_result f1, f2, f3, f4;
  int status = 0;
  status += gsl_sf_fact_e((ta + tb - tc)/2, &f1);
  status += gsl_sf_fact_e((ta + tc - tb)/2, &f2);
  status += gsl_sf_fact_e((tb + tc - ta)/2, &f3);
  status += gsl_sf_fact_e((ta + tb + tc)/2 + 1, &f4);
  if(status != 0) {
    OVERFLOW_ERROR(d);
  }
  d->val = f1.val * f2.val * f3.val / f4.val;
  d->err = 4.0 * GSL_DBL_EPSILON * fabs(d->val);
  return GSL_SUCCESS;
}


static
int
triangle_selection_fails(int two_ja, int two_jb, int two_jc)
{
  /*
   * enough to check the triangle condition for one spin vs. the other two
   */
  return ( (two_jb < abs(two_ja - two_jc)) || (two_jb > two_ja + two_jc) ||
           GSL_IS_ODD(two_ja + two_jb + two_jc) );
}


static
int
m_selection_fails(int two_ja, int two_jb, int two_jc,
                  int two_ma, int two_mb, int two_mc)
{
  return (
         abs(two_ma) > two_ja 
      || abs(two_mb) > two_jb
      || abs(two_mc) > two_jc
      || GSL_IS_ODD(two_ja + two_ma)
      || GSL_IS_ODD(two_jb + two_mb)
      || GSL_IS_ODD(two_jc + two_mc)
      || (two_ma + two_mb + two_mc) != 0
          );
}


/*-*-*-*-*-*-*-*-*-*-*-* Functions with Error Codes *-*-*-*-*-*-*-*-*-*-*-*/


int
gsl_sf_coupling_3j_e (int two_ja, int two_jb, int two_jc,
                      int two_ma, int two_mb, int two_mc,
                      gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(two_ja < 0 || two_jb < 0 || two_jc < 0) {
    DOMAIN_ERROR(result);
  }
  else if (   triangle_selection_fails(two_ja, two_jb, two_jc)
           || m_selection_fails(two_ja, two_jb, two_jc, two_ma, two_mb, two_mc)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else if ( two_ma == 0 && two_mb == 0 && two_mc == 0
            && ((two_ja + two_jb + two_jc) % 4 == 2) ) {
    /* Special case for (ja jb jc; 0 0 0) = 0 when ja+jb+jc=odd */
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    int jca  = (-two_ja + two_jb + two_jc) / 2,
        jcb  = ( two_ja - two_jb + two_jc) / 2,
        jcc  = ( two_ja + two_jb - two_jc) / 2,
        jmma = ( two_ja - two_ma) / 2,
        jmmb = ( two_jb - two_mb) / 2,
        jmmc = ( two_jc - two_mc) / 2,
        jpma = ( two_ja + two_ma) / 2,
        jpmb = ( two_jb + two_mb) / 2,
        jpmc = ( two_jc + two_mc) / 2,
        jsum = ( two_ja + two_jb + two_jc) / 2,
        kmin = locMax3 (0, jpmb - jmmc, jmma - jpmc),
        kmax = locMin3 (jcc, jmma, jpmb),
        k, sign = GSL_IS_ODD (kmin - jpma + jmmb) ? -1 : 1,
        status = 0;
    double sum_pos = 0.0, sum_neg = 0.0, sum_err = 0.0;
    gsl_sf_result bc1, bc2, bc3, bcn1, bcn2, bcd1, bcd2, bcd3, bcd4, term, lnorm;

    status += gsl_sf_lnchoose_e (two_ja, jcc , &bcn1);
    status += gsl_sf_lnchoose_e (two_jb, jcc , &bcn2);
    status += gsl_sf_lnchoose_e (jsum+1, jcc , &bcd1);
    status += gsl_sf_lnchoose_e (two_ja, jmma, &bcd2);
    status += gsl_sf_lnchoose_e (two_jb, jmmb, &bcd3);
    status += gsl_sf_lnchoose_e (two_jc, jpmc, &bcd4);

    lnorm.val = 0.5 * (bcn1.val + bcn2.val - bcd1.val - bcd2.val - bcd3.val
                       - bcd4.val - log(two_jc + 1.0));
    lnorm.err = 0.5 * (bcn1.err + bcn2.err + bcd1.err + bcd2.err + bcd3.err
                       + bcd4.err + GSL_DBL_EPSILON * log(two_jc + 1.0));

    for (k = kmin; k <= kmax; k++) {
      status += gsl_sf_lnchoose_e (jcc, k, &bc1);
      status += gsl_sf_lnchoose_e (jcb, jmma - k, &bc2);
      status += gsl_sf_lnchoose_e (jca, jpmb - k, &bc3);
      status += gsl_sf_exp_err_e(bc1.val + bc2.val + bc3.val + lnorm.val,
                                 bc1.err + bc2.err + bc3.err + lnorm.err, 
                                 &term);
      
      if (status != 0) {
        OVERFLOW_ERROR (result);
      }
      
      if (sign < 0) {
        sum_neg += term.val;
      } else {
        sum_pos += term.val;
      }

      sum_err += term.err;

      sign = -sign;
    }
    
    result->val  = sum_pos - sum_neg;
    result->err  = sum_err;
    result->err += 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += 2.0 * GSL_DBL_EPSILON * (kmax - kmin) * fabs(result->val);

    return GSL_SUCCESS;
  }
}


int
gsl_sf_coupling_6j_e(int two_ja, int two_jb, int two_jc,
                     int two_jd, int two_je, int two_jf,
                     gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(   two_ja < 0 || two_jb < 0 || two_jc < 0
     || two_jd < 0 || two_je < 0 || two_jf < 0
     ) {
    DOMAIN_ERROR(result);
  }
  else if(   triangle_selection_fails(two_ja, two_jb, two_jc)
          || triangle_selection_fails(two_ja, two_je, two_jf)
          || triangle_selection_fails(two_jb, two_jd, two_jf)
          || triangle_selection_fails(two_je, two_jd, two_jc)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    gsl_sf_result n1;
    gsl_sf_result d1, d2, d3, d4, d5, d6;
    double norm;
    int tk, tkmin, tkmax;
    double phase;
    double sum_pos = 0.0;
    double sum_neg = 0.0;
    double sumsq_err = 0.0;
    int status = 0;
    status += delta(two_ja, two_jb, two_jc, &d1);
    status += delta(two_ja, two_je, two_jf, &d2);
    status += delta(two_jb, two_jd, two_jf, &d3);
    status += delta(two_je, two_jd, two_jc, &d4);
    if(status != GSL_SUCCESS) {
      OVERFLOW_ERROR(result);
    }
    norm = sqrt(d1.val) * sqrt(d2.val) * sqrt(d3.val) * sqrt(d4.val);
    
    tkmin = locMax3(0,
                   two_ja + two_jd - two_jc - two_jf,
                   two_jb + two_je - two_jc - two_jf);

    tkmax = locMin5(two_ja + two_jb + two_je + two_jd + 2,
                    two_ja + two_jb - two_jc,
                    two_je + two_jd - two_jc,
                    two_ja + two_je - two_jf,
                    two_jb + two_jd - two_jf);

    phase = GSL_IS_ODD((two_ja + two_jb + two_je + two_jd + tkmin)/2)
            ? -1.0
            :  1.0;

    for(tk=tkmin; tk<=tkmax; tk += 2) {
      double term;
      double term_err;
      gsl_sf_result den_1, den_2;
      gsl_sf_result d1_a, d1_b;
      status = 0;

      status += gsl_sf_fact_e((two_ja + two_jb + two_je + two_jd - tk)/2 + 1, &n1);
      status += gsl_sf_fact_e(tk/2, &d1_a);
      status += gsl_sf_fact_e((two_jc + two_jf - two_ja - two_jd + tk)/2, &d1_b);
      status += gsl_sf_fact_e((two_jc + two_jf - two_jb - two_je + tk)/2, &d2);
      status += gsl_sf_fact_e((two_ja + two_jb - two_jc - tk)/2, &d3);
      status += gsl_sf_fact_e((two_je + two_jd - two_jc - tk)/2, &d4);
      status += gsl_sf_fact_e((two_ja + two_je - two_jf - tk)/2, &d5);
      status += gsl_sf_fact_e((two_jb + two_jd - two_jf - tk)/2, &d6);

      if(status != GSL_SUCCESS) {
        OVERFLOW_ERROR(result);
      }

      d1.val = d1_a.val * d1_b.val;
      d1.err = d1_a.err * fabs(d1_b.val) + fabs(d1_a.val) * d1_b.err;

      den_1.val  = d1.val*d2.val*d3.val;
      den_1.err  = d1.err * fabs(d2.val*d3.val);
      den_1.err += d2.err * fabs(d1.val*d3.val);
      den_1.err += d3.err * fabs(d1.val*d2.val);

      den_2.val  = d4.val*d5.val*d6.val;
      den_2.err  = d4.err * fabs(d5.val*d6.val);
      den_2.err += d5.err * fabs(d4.val*d6.val);
      den_2.err += d6.err * fabs(d4.val*d5.val);

      term  = phase * n1.val / den_1.val / den_2.val;
      phase = -phase;
      term_err  = n1.err / fabs(den_1.val) / fabs(den_2.val);
      term_err += fabs(term / den_1.val) * den_1.err;
      term_err += fabs(term / den_2.val) * den_2.err;

      if(term >= 0.0) {
        sum_pos += norm*term;
      }
      else {
        sum_neg -= norm*term;
      }

      sumsq_err += norm*norm * term_err*term_err;
    }

    result->val  = sum_pos - sum_neg;
    result->err  = 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += sqrt(sumsq_err / (0.5*(tkmax-tkmin)+1.0));
    result->err += 2.0 * GSL_DBL_EPSILON * (tkmax - tkmin + 2.0) * fabs(result->val);

    return GSL_SUCCESS;
  }
}

int
gsl_sf_coupling_9j_e(int two_ja, int two_jb, int two_jc,
                     int two_jd, int two_je, int two_jf,
                     int two_jg, int two_jh, int two_ji,
                     gsl_sf_result * result)
{
  /* CHECK_POINTER(result) */

  if(   two_ja < 0 || two_jb < 0 || two_jc < 0
     || two_jd < 0 || two_je < 0 || two_jf < 0
     || two_jg < 0 || two_jh < 0 || two_ji < 0
     ) {
    DOMAIN_ERROR(result);
  }
  else if(   triangle_selection_fails(two_ja, two_jb, two_jc)
          || triangle_selection_fails(two_jd, two_je, two_jf)
          || triangle_selection_fails(two_jg, two_jh, two_ji)
          || triangle_selection_fails(two_ja, two_jd, two_jg)
          || triangle_selection_fails(two_jb, two_je, two_jh)
          || triangle_selection_fails(two_jc, two_jf, two_ji)
     ) {
    result->val = 0.0;
    result->err = 0.0;
    return GSL_SUCCESS;
  }
  else {
    int tk;
    int tkmin = locMax3(abs(two_ja-two_ji), abs(two_jh-two_jd), abs(two_jb-two_jf));
    int tkmax = locMin3(two_ja + two_ji, two_jh + two_jd, two_jb + two_jf);
    double sum_pos = 0.0;
    double sum_neg = 0.0;
    double sumsq_err = 0.0;
    double phase;
    for(tk=tkmin; tk<=tkmax; tk += 2) {
      gsl_sf_result s1, s2, s3;
      double term;
      double term_err;
      int status = 0;

      status += gsl_sf_coupling_6j_e(two_ja, two_ji, tk,  two_jh, two_jd, two_jg,  &s1);
      status += gsl_sf_coupling_6j_e(two_jb, two_jf, tk,  two_jd, two_jh, two_je,  &s2);
      status += gsl_sf_coupling_6j_e(two_ja, two_ji, tk,  two_jf, two_jb, two_jc,  &s3);

      if(status != GSL_SUCCESS) {
        OVERFLOW_ERROR(result);
      }
      term = s1.val * s2.val * s3.val;
      term_err  = s1.err * fabs(s2.val*s3.val);
      term_err += s2.err * fabs(s1.val*s3.val);
      term_err += s3.err * fabs(s1.val*s2.val);

      if(term >= 0.0) {
        sum_pos += (tk + 1) * term;
      }
      else {
        sum_neg -= (tk + 1) * term;
      }

      sumsq_err += ((tk+1) * term_err) * ((tk+1) * term_err);
    }

    phase = GSL_IS_ODD(tkmin) ? -1.0 : 1.0;

    result->val  = phase * (sum_pos - sum_neg);
    result->err  = 2.0 * GSL_DBL_EPSILON * (sum_pos + sum_neg);
    result->err += sqrt(sumsq_err / (0.5*(tkmax-tkmin)+1.0));
    result->err += 2.0 * GSL_DBL_EPSILON * (tkmax-tkmin + 2.0) * fabs(result->val);

    return GSL_SUCCESS;
  }
}


/*-*-*-*-*-*-*-*-*-* Functions w/ Natural Prototypes *-*-*-*-*-*-*-*-*-*-*/

#define EVAL_RESULT(fn) \
   gsl_sf_result result; \
   int status = fn; \
   if (status != GSL_SUCCESS) { \
     printf("GSL error\n"); exit(1); \
   } ; \
   return result.val;


double gsl_sf_coupling_3j(int two_ja, int two_jb, int two_jc,
                          int two_ma, int two_mb, int two_mc)
{
  EVAL_RESULT(gsl_sf_coupling_3j_e(two_ja, two_jb, two_jc,
                                   two_ma, two_mb, two_mc,
                                   &result));
}

double gsl_sf_coupling_6j(int two_ja, int two_jb, int two_jc,
                          int two_jd, int two_je, int two_jf)
{
  EVAL_RESULT(gsl_sf_coupling_6j_e(two_ja, two_jb, two_jc,
                                   two_jd, two_je, two_jf,
                                   &result));
}

double gsl_sf_coupling_9j(int two_ja, int two_jb, int two_jc,
                          int two_jd, int two_je, int two_jf,
                          int two_jg, int two_jh, int two_ji)
{
  EVAL_RESULT(gsl_sf_coupling_9j_e(two_ja, two_jb, two_jc,
                                   two_jd, two_je, two_jf,
                                   two_jg, two_jh, two_ji,
                                   &result));
}
