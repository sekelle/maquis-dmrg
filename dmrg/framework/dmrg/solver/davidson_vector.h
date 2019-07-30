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

#ifndef DAVIDSON_VECTOR
#define DAVIDSON_VECTOR

#include <vector>


template <class T>
class DavidsonVector
{
public:
    typedef T value_type;
    typedef T real_type;
    typedef double magnitude_type;

    DavidsonVector();
    DavidsonVector(std::vector<const T*> const& bm,
                   std::vector<std::size_t> block_sizes);
    DavidsonVector(std::vector<std::size_t> block_sizes);

    DavidsonVector(DavidsonVector const&);

    DavidsonVector& operator=(DavidsonVector);

    size_t num_elements() const;

    T*       operator[](size_t b);
    const T* operator[](size_t b) const;

    void clear();

    std::vector<T*>& data_view();
    std::vector<const T*> data_view() const;

    std::vector<std::size_t> const& blocks() const;

    DavidsonVector const& operator*=(const value_type);
    DavidsonVector const& operator/=(const value_type);

    DavidsonVector const& operator+=(DavidsonVector const&);
    DavidsonVector const& operator-=(DavidsonVector const&);

    real_type scalar_norm() const;
    value_type scalar_overlap(DavidsonVector const&) const;

    void swap_with(DavidsonVector& other);
    template <class T_>
    friend void swap(DavidsonVector<T_>& a, DavidsonVector<T_>& b);

private:
    std::vector<T> buffer;
    std::vector<T*> view;
    std::vector<std::size_t> block_sizes;

    void create_view();
};


template <class T>
DavidsonVector<T> operator*(T scal, DavidsonVector<T> const& rhs)
{
    DavidsonVector<T> ret = rhs;
    return ret*=scal;
}
template <class T>
DavidsonVector<T> operator*(DavidsonVector<T> const& rhs, T scal)
{
    DavidsonVector<T> ret = rhs;
    return ret*=scal;
}
template <class T>
DavidsonVector<T> operator/(T scal, DavidsonVector<T> const& rhs)
{
    DavidsonVector<T> ret = rhs;
    return ret/=scal;
}
template <class T>
DavidsonVector<T> operator/(DavidsonVector<T> const& rhs, T scal)
{
    DavidsonVector<T> ret = rhs;
    return ret/=scal;
}

template <class T>
DavidsonVector<T> operator+(DavidsonVector<T> const& a, DavidsonVector<T> const& b)
{
    DavidsonVector<T> ret = a;
    return ret+=b;
}
template <class T>
DavidsonVector<T> operator-(DavidsonVector<T> const& a, DavidsonVector<T> const& b)
{
    DavidsonVector<T> ret = a;
    return ret-=b;
}
template <class T>
DavidsonVector<T> operator-(DavidsonVector<T> const& a)
{
    DavidsonVector<T> ret = a;
    return ret*=-1.0;
}

// explicit instatiation, refer to the .cpp file
extern template class DavidsonVector<double>;

#endif
