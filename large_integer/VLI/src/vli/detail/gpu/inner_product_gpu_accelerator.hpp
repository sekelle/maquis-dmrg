/*
 *Very Large Integer Library, License - Version 1.0 - May 3rd, 2012
 *
 *Timothee Ewart - University of Geneva,
 *Andreas Hehn - Swiss Federal Institute of technology Zurich.
 *Maxim Milakov - NVIDIA
 *
 *Permission is hereby granted, free of charge, to any person or organization
 *obtaining a copy of the software and accompanying documentation covered by
 *this license (the "Software") to use, reproduce, display, distribute,
 *execute, and transmit the Software, and to prepare derivative works of the
 *Software, and to permit third-parties to whom the Software is furnished to
 *do so, all subject to the following:
 *
 *The copyright notices in the Software and this entire statement, including
 *the above license grant, this restriction and the following disclaimer,
 *must be included in all copies of the Software, in whole or in part, and
 *all derivative works of the Software, unless such copies or derivative
 *works are solely in the form of machine-executable object code generated by
 *a source language processor.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 *SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 *FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 *ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *DEALINGS IN THE SOFTWARE.
 
 */

#ifndef INNER_PRODUCT_GPU_BOOSTER_HPP
#define INNER_PRODUCT_GPU_BOOSTER_HPP
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vli/detail/gpu/detail/gpu_error_message.h"
#include "utils/timings.h"

#include "vli/detail/kernels_gpu.h"
#include "vli/detail/gpu/detail/variables_gpu.h"

namespace vli
{
 
    // a lot of forward declaration    
    template <class Coeff, class MaxOrder, class Var0, class Var1, class Var2, class Var3>
    class polynomial;

    template <class polynomial>
    class vector_polynomial;
   
    template <class VectorPolynomial>
    struct inner_product_result_type;
   
    template <class Coeff, class MaxOrder, class Var0, class Var1, class Var2, class Var3>
    struct inner_product_result_type< vector_polynomial<polynomial<Coeff,MaxOrder,Var0,Var1,Var2,Var3> > >; 

    namespace detail {

        // this class helper distinguishes keep_order and max_order polynomial by template specialization
       // template <class Coeff, class MaxOrder, class Var0, class Var1, class Var2, class Var3>
        template <class polynomial>
        struct inner_product_gpu_helper{
        };

        template <class Coeff,  int  Order, class Var0, class Var1, class Var2, class Var3>
        struct inner_product_gpu_helper<polynomial<Coeff, max_order_each<Order>, Var0, Var1, Var2, Var3> >{
            static inline typename inner_product_result_type<vector_polynomial<polynomial<Coeff,max_order_each<Order>, Var0, Var1, Var2, Var3> > >::type /* return type ~~'*/
            inner_product_gpu(
                 vector_polynomial<polynomial<Coeff,max_order_each<Order>, Var0, Var1, Var2, Var3> > const& v1,
                 vector_polynomial<polynomial<Coeff,max_order_each<Order>, Var0, Var1, Var2, Var3> > const& v2
            ) {
            assert(v1.size() == v2.size());
            std::size_t size_v = v1.size();
          
            #ifdef _OPENMP
                std::vector<typename inner_product_result_type<vector_polynomial<polynomial<Coeff, max_order_each<Order>, Var0, Var1, Var2, Var3> > >::type > res(omp_get_max_threads());
            #else
                typename inner_product_result_type<vector_polynomial<polynomial<Coeff,max_order_each<Order>, Var0, Var1, Var2, Var3> > >::type res;
            #endif
          
            typename inner_product_result_type<vector_polynomial<polynomial<Coeff, max_order_each<Order>, Var0, Var1, Var2, Var3> > >::type poly;
          
            std::size_t split = static_cast<std::size_t>(VLI_SPLIT_PARAM*v1.size());
            vli::detail::gpu_inner_product_vector<Coeff::numbits, max_order_each<Order>, num_of_variables_helper<Var0, Var1, Var2, Var3>::value >(split, &v1[0](0,0)[0], &v2[0](0,0)[0]);
          
            #pragma omp parallel for schedule(dynamic)
            for(std::size_t i=split ; i < size_v ; ++i){
                #ifdef _OPENMP
                   res[omp_get_thread_num()] += v1[i]*v2[i]; //local reduction specific for every thread
                #else
                   res += v1[i]*v2[i];
                #endif
            }
          
            #ifdef _OPENMP //final omp reduction
            for(int i=1; i < omp_get_max_threads(); ++i)
                res[0]+=res[i];
            #endif

            gpu::cu_check_error(cudaMemcpy((void*)&poly(0,0),(void*)gpu_get_polynomial(),
                               2*Coeff::numwords*result_stride<0,num_of_variables_helper<Var0, Var1, Var2, Var3>::value, max_order_each<Order>::value>::value
                                                *result_stride<1,num_of_variables_helper<Var0, Var1, Var2, Var3>::value, max_order_each<Order>::value>::value
                                                *result_stride<2,num_of_variables_helper<Var0, Var1, Var2, Var3>::value, max_order_each<Order>::value>::value
                                                *result_stride<3,num_of_variables_helper<Var0, Var1, Var2, Var3>::value, max_order_each<Order>::value>::value
                                                *sizeof(long),cudaMemcpyDeviceToHost),__LINE__);// this thing synchronizes 
          
            #ifdef _OPENMP
                res[0] += poly;
                return res[0];
            #else
                res += poly;
                return res;
            #endif
            } // end function
        }; // end specialization class helper

        template <class Coeff, int Order, class Var0, class Var1, class Var2, class Var3>
        struct inner_product_gpu_helper<polynomial<Coeff, max_order_combined<Order>, Var0, Var1, Var2, Var3> >{
        static inline typename inner_product_result_type<vector_polynomial<polynomial<Coeff,max_order_combined<Order>,Var0,Var1,Var2,Var3> > >::type
            inner_product_gpu(
                 vector_polynomial<polynomial<Coeff,max_order_combined<Order>,Var0,Var1,Var2,Var3> > const& v1,
                 vector_polynomial<polynomial<Coeff,max_order_combined<Order>,Var0,Var1,Var2,Var3> > const& v2
            ) {
                assert(v1.size() == v2.size());
                std::size_t size_v = v1.size();
              
                #ifdef _OPENMP
                    std::vector<typename inner_product_result_type<vector_polynomial<polynomial<Coeff, max_order_combined<Order>, Var0, Var1, Var2, Var3> > >::type > res(omp_get_max_threads());
                #else
                    typename inner_product_result_type<vector_polynomial<polynomial<Coeff, max_order_combined<Order>, Var0, Var1, Var2, Var3> > >::type res;
                #endif
            
                typename inner_product_result_type<vector_polynomial<polynomial<Coeff, max_order_combined<Order>, Var0, Var1, Var2, Var3> > >::type poly;
              
                std::size_t split = static_cast<std::size_t>(VLI_SPLIT_PARAM*v1.size());
                vli::detail::gpu_inner_product_vector<Coeff::numbits, max_order_combined<Order>, num_of_variables_helper<Var0, Var1, Var2, Var3>::value >(split, &v1[0](0,0)[0], &v2[0](0,0)[0]);
            
                #pragma omp parallel for schedule(dynamic)
                for(std::size_t i=split ; i < size_v ; ++i){
                    #ifdef _OPENMP
                       res[omp_get_thread_num()] += v1[i]*v2[i]; //local reduction specific for every thread
                    #else
                       res += v1[i]*v2[i];
                    #endif
                }
              
                #ifdef _OPENMP //final omp reduction
                for(int i=1; i < omp_get_max_threads(); ++i)
                    res[0]+=res[i];
                #endif
                
                gpu::cu_check_error(cudaMemcpy((void*)&poly(0,0),(void*)gpu_get_polynomial(),
                                                2*Coeff::numwords*max_order_combined_helpers::size<num_of_variables_helper<Var0,Var1,Var2,Var3 >::value+1, 2*Order>::value
                                                *sizeof(long),cudaMemcpyDeviceToHost),__LINE__);// this thing synchronizes 
                               
                #ifdef _OPENMP
                    res[0] += poly;
                    return res[0];
                #else
                    res += poly;
                    return res;
                #endif
            }
        };

    } // end namespace detail
} // end namespace vli

#endif //INNER_PRODUCT_GPU_BOOSTER_HPP
