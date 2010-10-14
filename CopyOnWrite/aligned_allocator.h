#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <vector>
#include <iostream>

#include <boost/type_traits/aligned_storage.hpp>

template<class T>
class aligned_allocator : public std::allocator<T>
{
protected:
	typedef std::allocator<T> alloc_t;
	
public:
	typedef typename alloc_t::pointer pointer;
	typedef typename alloc_t::const_pointer const_pointer;
	typedef typename alloc_t::reference reference;
	typedef typename alloc_t::const_reference const_reference;
	
	typedef typename alloc_t::value_type value_type;
	typedef typename alloc_t::size_type size_type;
	typedef typename alloc_t::difference_type difference_type;
	
	pointer allocate(size_type n) const
	{
	    // static_cast unfortunately doesn't work
        return reinterpret_cast<pointer>(new typename boost::aligned_storage<sizeof(T), 16>::type[n]);
	}
	void deallocate(pointer p, size_type n) const
	{
        delete[] p;
	}
	
	template <typename T2>
	struct rebind
	{
	   typedef aligned_allocator<T2> other;
	};
};

#endif
