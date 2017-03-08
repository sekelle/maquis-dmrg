/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2015 Institute for Theoretical Physics, ETH Zurich
 *               2015-2015 by Sebastian Keller <sebkelle@phys.ethz.ch>
 * 
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef GSL_COUPLING_H
#define GSL_COUPLING_H

extern "C" {
    double gsl_sf_coupling_3j(int two_ja, int two_jb, int two_jc, int two_ma, int two_mb, int two_mc);
    double gsl_sf_coupling_6j(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf);
    double gsl_sf_coupling_9j(int two_ja, int two_jb, int two_jc, int two_jd, int two_je, int two_jf, int two_jg, int two_jh, int two_ji);
}

namespace SU2 {

    inline double mod_coupling(int a, int b, int c,
                        int d, int e, int f,
                        int g, int h, int i)
    {
        double ret = sqrt( (g+1.) * (h+1.) * (c+1.) * (f+1.) ) *
               gsl_sf_coupling_9j(a, b, c,
                                  d, e, f,
                                  g, h, i);
        return ret;
    }

    inline bool triangle(int a, int b, int c)
    {
        return std::abs(a-b) <= c && c <= a+b;
    }

    template <class T>
    inline void set_coupling(int a, int b, int c,
                             int d, int e, int f,
                             int g, int h, int i, T init, T couplings[])
    {
        couplings[0] = 0.0;
        couplings[1] = 0.0;
        couplings[2] = 0.0;
        couplings[3] = 0.0;
        T prefactor = T(sqrt((i+1.)*(a+1.)/((g+1.)*(c+1.)))) * init;
        if (triangle(a,b,c))
        {
            couplings[0] = prefactor * (T)::SU2::mod_coupling(a, b, c, d, e, f, g, h, i);
            couplings[2] = prefactor * (T)::SU2::mod_coupling(a, b, c, d, e, f, g, 2, i);
        }
        if (triangle(a,2,c))
        {
            couplings[1] = prefactor * (T)::SU2::mod_coupling(a, 2, c, d, e, f, g, h, i);
            couplings[3] = prefactor * (T)::SU2::mod_coupling(a, 2, c, d, e, f, g, 2, i);
        }
    }

    template <typename T>
    class Wigner9jCacheI
    {
    public:
        Wigner9jCacheI(int J, int I, int Ip)
        {
            int two_sp = std::abs(I-Ip);

            Jpmax = J + 2;
            Jpmin = ((J-2) < 0) ? 0 : J-2;
            int nj = Jpmax - Jpmin + 1;

            coefficients.resize(12 * 4 * nj);

            //std::cout << "Jpmin " << Jpmin << " Jpmax " << Jpmax << std::endl;
            for (int Jp = Jpmin; Jp <= Jpmax; ++Jp)
            {
                int two_s = std::abs(J-Jp);

                int K[12], Ap[12];

                //(A == 0)
                    K[0] = 0; Ap[0] = 0;
                    K[1] = 1; Ap[1] = 1;
                    K[2] = 2; Ap[2] = 2;
                    K[3] = 0; Ap[3] = 0;
                //(A == 1) 
                    K[4+0] = 0; Ap[4+0] = 1;
                    K[4+1] = 1; Ap[4+1] = 0;
                    K[4+2] = 2; Ap[4+2] = 1;
                    K[4+3] = 1; Ap[4+3] = 2;
                //(A == 2)
                    K[8+0] = 0; Ap[8+0] = 2;
                    K[8+1] = 1; Ap[8+1] = 1;
                    K[8+2] = 2; Ap[8+2] = 0;
                    K[8+3] = 0; Ap[8+3] = 0;

                //std::cout << "  Jp" << Jp << std::endl;
                for (int i = 0; i < 12; ++i)
                {
                    int A = i/4;
                    //std::cout << "    " << i << " " << J << two_s << Jp << " " << A << K[i] << Ap[i] << " " << I << two_sp << Ip << std::endl;
                    int offset = 4*i + 48 * (Jp-Jpmin);
                    set_coupling(J, two_s, Jp, A, K[i], Ap[i], I, two_sp, Ip, T(1), &coefficients[offset]);
                }
            }
        }

        void set_scale(int A, int K, int Ap, int Jp, T scale, T couplings[])
        {
            //std::cout << "hash " << A << K << Ap << " " << hash(A,K,Ap) << std::endl;
            T* c2 = &coefficients[4*hash(A,K,Ap) + 48 * (Jp-Jpmin)];
            std::transform(c2, c2+4, couplings, boost::lambda::_1*scale);
        }

    private:

        static int hash(int a, int b, int c)
        {
            return 4*a + b + ((a==1 && b==1) ? c : 0);
        }

        int Jpmax;
        int Jpmin;

        std::vector<T> coefficients;
    };

    template <typename T, class SymmGroup>
    class Wigner9jCache : public Wigner9jCacheI<T>
    {
        typedef typename SymmGroup::charge charge;
        typedef Wigner9jCacheI<T> base;

    public:

        Wigner9jCache(charge lc, charge mc, charge rc)
            : base(SymmGroup::spin(mc), SymmGroup::spin(lc), SymmGroup::spin(rc)) {}
    };
}

#endif
