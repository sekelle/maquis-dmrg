#ifndef MAQUIS_TASKS_NUMERIC_H
#define MAQUIS_TASKS_NUMERIC_H

#include <malloc.h>


template <class T>
inline void mydaxpy(std::size_t n, T a, const T* x, T* y)
{
    std::cout << "Generic\n";
    std::transform(x, x+n, y, y, boost::lambda::_1*a+boost::lambda::_2);
}

inline void mydaxpy(std::size_t n, double a, const double* x, double* y)
{
  // broadcast the scale factor into a register
  __m256d x0 = _mm256_broadcast_sd(&a);

  // align
  //std::size_t xv = *reinterpret_cast<std::size_t*>(&x);
  //std::size_t yv = *reinterpret_cast<std::size_t*>(&y);
  assert((uintptr_t)(x) % 32 == 0);
  assert((uintptr_t)(y) % 32 == 0);

  std::size_t ndiv4 = n/4;

  for (int i=0; i<ndiv4; ++i) {
    __m256d x1 = _mm256_load_pd(x+4*i);
    __m256d x2 = _mm256_load_pd(y+4*i);
    __m256d x3 = _mm256_mul_pd(x0, x1);
    __m256d x4 = _mm256_add_pd(x2, x3);
    _mm256_store_pd(y+4*i, x4);
  }

  for (int i=ndiv4*4; i < n ; ++i)
    y[i] += a*x[i];
}


template <class T>
inline void blas_dgemm(const T* A, const T* B, T* C, int M, int K, int N, bool trA)
{
    throw std::runtime_error("gemm not implemented\n");
}

inline void blas_dgemm(const double* A, const double* B, double* C, int M, int K, int N, bool trA)
{
    double one=1;
    char trans = 'T';
    char notrans = 'N';
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
            //mydaxpy(m_size * r_size, tasks[i][j].scale, &T[tasks[i][j].t_index](0,0), &S(0,0));
        }

        blas_dgemm(left[i], s_buffer, out, ls, ms, rs, transL[i]);
    }

    free(s_buffer);
}

#endif
