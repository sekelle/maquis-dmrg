#ifndef __MAQUIS_TYPES_TRAITS_HPP__
#define __MAQUIS_TYPES_TRAITS_HPP__

#include <alps/numeric/matrix/matrix_traits.hpp> 

namespace maquis { namespace traits {

    template<class T> struct scalar_type { typedef typename T::value_type type; };
    template<class T> struct real_type { typedef typename real_type<typename T::value_type>::type type; };
    template<>        struct real_type<double> { typedef double type; };
    template<>        struct real_type<float> { typedef float type; };
    template<class T> struct real_type<std::complex<T> > { typedef T type; };
    template<class T> struct real_identity { static const T value; };
    template<class T> struct imag_identity { static const T value; };
    template<class T> struct real_identity<std::complex<T> > { static const std::complex<T> value; };
    template<class T> struct imag_identity<std::complex<T> > { static const std::complex<T> value; };
    template<class T> const T real_identity<T>::value = 1;
    template<class T> const T imag_identity<T>::value = 1;
    template<class T> const std::complex<T> real_identity<std::complex<T> >::value = std::complex<T>(1,0);
    template<class T> const std::complex<T> imag_identity<std::complex<T> >::value = std::complex<T>(0,1);

    template <class Matrix> struct transpose_view { typedef Matrix type; };

    namespace detail {

        template <class Base, class Arg> struct SwapArg {};

        template< template<typename> class Base, class T, class Arg>
        struct SwapArg<Base<T>, Arg>
        {
            typedef Base<Arg> type;
        };

        template <class Base, class NewArg> struct SwapSecondArg {};

        template< template<typename, typename> class Base, class A1, class A2, class NewArg>
        struct SwapSecondArg<Base<A1, A2>, NewArg>
        {
            typedef Base<A1, NewArg> type;
        };

    }

    template<class Matrix, template<typename, unsigned> class Allocator, unsigned Alignment>
    struct aligned_matrix { };

    template < template<typename, typename> class Matrix, class T, class MemoryBlock,
               template<typename, unsigned> class Allocator, unsigned Alignment>
    struct aligned_matrix<Matrix<T, MemoryBlock>, Allocator, Alignment>
    {
        typedef typename detail::SwapSecondArg<MemoryBlock, Allocator<T, Alignment> >::type NewMemoryBlock;
        typedef Matrix<T, NewMemoryBlock> type;
    };

} // namespace traits
} // namespace maquis

#endif
