/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2019 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2019-2019 by Sebastian Keller <sebkelle@ethz.ch>
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

#include <vector>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>

#include "dmrg/utils/utils.hpp"
#include "dmrg/utils/aligned_allocator.hpp"

#include "dmrg/solver/accelerator.h"
#include "dmrg/solver/numeric/gpu.h"

#include "davidson_vector.h"

//template <class T>
//DavidsonVector<T> operator*(T scal, DavidsonVector<T> const& rhs)
//{
//    DavidsonVector<T> ret = rhs;
//    return ret*=scal;
//}
//template <class T>
//DavidsonVector<T> operator*(DavidsonVector<T> const& rhs, T scal)
//{
//    DavidsonVector<T> ret = rhs;
//    return ret*=scal;
//}
//template <class T>
//DavidsonVector<T> operator/(T scal, DavidsonVector<T> const& rhs)
//{
//    DavidsonVector<T> ret = rhs;
//    return ret/=scal;
//}
//template <class T>
//DavidsonVector<T> operator/(DavidsonVector<T> const& rhs, T scal)
//{
//    DavidsonVector<T> ret = rhs;
//    return ret/=scal;
//}
//
//template <class T>
//DavidsonVector<T> operator+(DavidsonVector<T> const& a, DavidsonVector<T> const& b)
//{
//    DavidsonVector<T> ret = a;
//    return ret+=b;
//}
//template <class T>
//DavidsonVector<T> operator-(DavidsonVector<T> const& a, DavidsonVector<T> const& b)
//{
//    DavidsonVector<T> ret = a;
//    return ret-=b;
//}
//template <class T>
//DavidsonVector<T> operator-(DavidsonVector<T> const& a)
//{
//    DavidsonVector<T> ret = a;
//    return ret*=-1.0;
//}


template <class T>
DavidsonVector<T>::DavidsonVector() {}

template <class T>
DavidsonVector<T>::DavidsonVector(std::vector<const T*> const& bm,
                   std::vector<std::size_t> block_sizes_)
    : block_sizes(block_sizes_)
{
    size_t sz = 0;
    for (size_t b = 0; b < block_sizes.size(); ++b)
        sz += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_sizes[b]);

    buffer = std::vector<T>(sz);
    
    create_view(); 

    for (size_t b = 0; b < block_sizes.size(); ++b)
        std::copy(bm[b], bm[b]+block_sizes[b], view[b]);
}

template <class T>
DavidsonVector<T>::DavidsonVector(std::vector<std::size_t> block_sizes_)
    : block_sizes(block_sizes_)
{
    size_t sz = 0;
    for (size_t b = 0; b < block_sizes.size(); ++b)
        sz += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_sizes[b]);

    buffer = std::vector<T>(sz);
    
    create_view(); 
}

template <class T>
DavidsonVector<T>::DavidsonVector(DavidsonVector const& other)
{
    buffer = other.buffer;
    block_sizes = other.block_sizes;
    create_view();
}

template <class T>
DavidsonVector<T>& DavidsonVector<T>::operator=(DavidsonVector rhs)
{
    this->swap_with(rhs);
    return *this;
}



template <class T>
size_t DavidsonVector<T>::num_elements() const {
    return buffer.size();
}

template <class T>
void DavidsonVector<T>::create_view()
{
    view.resize(block_sizes.size());
    T* enumerator = buffer.data();
    for (size_t b = 0; b < block_sizes.size(); ++b)
    {
        view[b] = enumerator;
        enumerator += bit_twiddling::round_up<BUFFER_ALIGNMENT>(block_sizes[b]);
    }
}

template <class T>
T* DavidsonVector<T>::operator[](size_t b) { return view[b]; }

template <class T>
const T* DavidsonVector<T>::operator[](size_t b) const { return view[b]; }

template <class T>
void DavidsonVector<T>::clear() {
    buffer.clear();
    view.clear(); 
    block_sizes.clear();
}


template <class T>
std::vector<T*>& DavidsonVector<T>::data_view() {
    return view;
}

template <class T>
std::vector<const T*> DavidsonVector<T>::data_view() const {
    return std::vector<const T*>{begin(view), end(view)};
}

template <class T>
std::vector<std::size_t> const& DavidsonVector<T>::blocks() const {
    return block_sizes;
}


template <class T>
void DavidsonVector<T>::swap_with(DavidsonVector & other)
{
    swap(buffer, other.buffer);
    swap(view, other.view);
    swap(block_sizes, other.block_sizes);
}

template <class T>
void swap(DavidsonVector<T>& a, DavidsonVector<T>& b) { a.swap_with(b); }



template <class T>
DavidsonVector<T> const& DavidsonVector<T>::operator*=(value_type scal)
{
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] *= scal;

    return *this;
}

template <class T>
DavidsonVector<T> const& DavidsonVector<T>::operator/=(value_type scal)
{
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] /= scal;

    return *this;
}

template <class T>
DavidsonVector<T> const& DavidsonVector<T>::operator+=(DavidsonVector<T> const& rhs)
{
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] += rhs.buffer[i];

    return *this;
}

template <class T>
DavidsonVector<T> const& DavidsonVector<T>::operator-=(DavidsonVector<T> const& rhs)
{
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] -= rhs.buffer[i];

    return *this;
}


template <class T>
typename DavidsonVector<T>::real_type DavidsonVector<T>::scalar_norm() const
{
    
    real_type sum = 0;
    for (size_t i = 0; i < buffer.size(); ++i)
        sum += buffer[i] * buffer[i]; 

    return std::sqrt(sum);
}

template <class T>
typename DavidsonVector<T>::value_type DavidsonVector<T>::scalar_overlap(DavidsonVector<T> const& other) const
{
    
    real_type sum = 0;
    for (size_t i = 0; i < buffer.size(); ++i)
        sum += buffer[i] * other.buffer[i]; 

    return sum;
}

// explicit instantiation
template class DavidsonVector<double>;
