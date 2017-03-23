#ifndef MAQUIS_TASKS_NUMERIC_H
#define MAQUIS_TASKS_NUMERIC_H

#include <malloc.h>

template <class T>
inline void blas_dgemm(const T* A, const T* B, T* C, int M, int K, int N, bool trA)
{
    throw std::runtime_error("gemm not implemented\n");
}

inline void blas_dgemm(const double* A, const double* B, double* C, int M, int K, int N, bool trA)
{
    double one=1;
    char trans = 'N';
    char notrans = 'T';
    if (trA)
        dgemm_(&trans, &notrans, &M, &N, &K, &one, A, &K, B, &K, &one, C, &M);
    else
        dgemm_(&notrans, &notrans, &M, &N, &K, &one, A, &M, B, &K, &one, C, &M);
}


template <class T>
void dgemm_ddot(unsigned ls, unsigned ms, unsigned rs, unsigned b1size,
                unsigned* b2sz, bool* transL, unsigned ** tidx, T** alpha, const T** left, const T** t, T* out)
{
    typedef unsigned uint;

    uint t_size = ms * rs;

    T * s_buffer = (T*)memalign(32, t_size * sizeof(T));
    for (uint i = 0; i < b1size; ++i)
    {
        memset(s_buffer, 0, t_size * sizeof(T));
        T * alpha_i = alpha[i];
        unsigned * tidx_i = tidx[i];
        for (uint j = 0; j < b2sz[i]; ++j)
        {
            unsigned tpos = tidx_i[j];
            maquis::dmrg::detail::iterator_axpy(t[tpos], t[tpos] + t_size, s_buffer, alpha_i[j]);
        }

        blas_dgemm(left[i], s_buffer, out, ls, ms, rs, transL[i]);
    }

    free(s_buffer);
}

template <class T>
void daxpy_ddot(unsigned ms, unsigned rs, unsigned b2sz, T* alpha_i,
                unsigned * tidx_i, const T** t, T* out)
{
    typedef unsigned uint;
    uint t_size = ms * rs;

    for (uint j = 0; j < b2sz; ++j)
    {
        unsigned tpos = tidx_i[j];
        maquis::dmrg::detail::iterator_axpy(t[tpos], t[tpos] + t_size, out, alpha_i[j]);
    }
}

#endif
