//
// Copyright (c) 2002--2010
// Toon Knapen, Karl Meerbergen, Kresimir Fresl,
// Thomas Klimpel and Rutger ter Borg
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// Modified gemm wrapper to allow multiplying a part of the input matrices
//
//

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_LEVEL3_GEMM_OFFSET_HPP
#define BOOST_NUMERIC_BINDINGS_BLAS_LEVEL3_GEMM_OFFSET_HPP

#include <boost/numeric/bindings/blas/level3/gemm.hpp>

namespace boost {
namespace numeric {
namespace bindings {
namespace blas {

//  |---->a_offset               |---->b_offset           |---->c_offset
//         ---- ncol_a                  --- ncol_c               --- ncol_c
//  |-----|----|----|            |-----|---|----|         |-----|---|----|
//  |     |xxxx|    |            |     |xxx|    |         |     |xxx|    |
//  |     |xxxx|    |     *      |     |xxx|    |    =    |     |xxx|    |
//  |     |xxxx|    |            |     |xxx|    |         |     |xxx|    |
//  |-----|----|----|            |     |xxx|    |         |-----|---|----|
//                               |-----|---|----|


//                               // transb = true

//  |---->a_offset               |---->b_offset           |---->c_offset
//         ---- ncol_a                  ---- ncol_c               --- ncol_c
//  |-----|----|----|            |-----|----|----|         |-----|---|----|
//  |     |xxxx|    |            |     |xxxx|    |         |     |xxx|    |
//  |     |xxxx|    |     *      |     |xxxx|    |    =    |     |xxx|    |
//  |     |xxxx|    |            |     |xxxx|    |         |     |xxx|    |
//  |-----|----|----|            |-----|----|----|         |-----|---|----|
//                               

namespace detail {

template <class T, bool B>
struct pick { T operator()(T a, T b) { return a; } };

template <class T>
struct pick<T, false> { T operator()(T a, T b) { return b; } };

} // namespace detail

template< typename Value >
struct gemm_impl_offset {

    typedef Value value_type;
    typedef typename remove_imaginary< Value >::type real_type;
    typedef void result_type;

    template< typename MatrixA, typename MatrixB, typename MatrixC >
    static result_type invoke( const value_type alpha, const MatrixA& a,
            const MatrixB& b, const value_type beta, MatrixC& c,
            size_t a_offset, size_t b_offset, size_t c_offset,
            size_t ncol_a, size_t ncol_c) {
        namespace bindings = ::boost::numeric::bindings;
        typedef typename result_of::data_order< MatrixC >::type order;
        typedef typename result_of::trans_tag< MatrixB, order >::type transb;
        typedef typename result_of::trans_tag< MatrixA, order >::type transa;
        BOOST_STATIC_ASSERT( (is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixB >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (is_same< typename remove_const<
                typename bindings::value_type< MatrixA >::type >::type,
                typename remove_const< typename bindings::value_type<
                MatrixC >::type >::type >::value) );
        BOOST_STATIC_ASSERT( (bindings::is_mutable< MatrixC >::value) );
        BOOST_ASSERT( bindings::size_minor(a) == 1 ||
                bindings::stride_minor(a) == 1 );
        BOOST_ASSERT( bindings::size_minor(b) == 1 ||
                bindings::stride_minor(b) == 1 );
        BOOST_ASSERT( bindings::size_minor(c) == 1 ||
                bindings::stride_minor(c) == 1 );

        typedef detail::pick<size_t, boost::core::is_same<transa, tag::transpose>::value> picker_a;
        size_t  a_start_offset = picker_a()(a_offset * bindings::size_column(a), a_offset * bindings::size_row(a));

        typedef detail::pick<size_t, boost::core::is_same<transb, tag::transpose>::value> picker;
        size_t b_start_offset = picker()(b_offset * bindings::size_column(b), b_offset * bindings::size_row(b));

        detail::gemm( order(), transa(), transb(),
                bindings::size_row(c), ncol_c,
                ncol_a, alpha, bindings::begin_value(a) + a_start_offset,
                bindings::stride_major(a), bindings::begin_value(b) + b_start_offset,
                bindings::stride_major(b), beta, bindings::begin_value(c) + c_offset * bindings::size_row(c),
                bindings::stride_major(c) );
    }
};


template< typename MatrixA, typename MatrixB, typename MatrixC >
inline typename gemm_impl< typename bindings::value_type<
        MatrixA >::type >::result_type
  gemm( const typename bindings::value_type< MatrixA >::type alpha,
        const MatrixA& a, const MatrixB& b,
        const typename bindings::value_type< MatrixA >::type beta,
        MatrixC& c, size_t a_offset, size_t b_offset, size_t c_offset,
                    size_t ncol_a, size_t ncol_c) {
    gemm_impl_offset< typename bindings::value_type<
            MatrixA >::type >::invoke( alpha, a, b, beta, c,
                                       a_offset, b_offset, c_offset,
                                       ncol_a, ncol_c);
}

} // namespace blas
} // namespace bindings
} // namespace numeric
} // namespace boost

#endif
