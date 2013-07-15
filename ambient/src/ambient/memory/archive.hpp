/*
 * Ambient, License - Version 1.0 - May 3rd, 2012
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef AMBIENT_MEMORY_ARCHIVE
#define AMBIENT_MEMORY_ARCHIVE

#include <boost/archive/detail/common_iarchive.hpp>
#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#include <boost/archive/impl/archive_serializer_map.ipp>
#include <boost/serialization/vector.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <typeinfo>
#include <vector>
#include <memory.h>

namespace ambient { namespace memory {

    template<class Archive>
    class iarchive : public boost::archive::detail::common_iarchive< Archive > {
        public: friend class boost::archive::load_access;
    
        template<class T> void load_override(T& t, int){
            boost::archive::load(*this, t);
        }
        template<class T, class Allocator> void load_override(std::vector<T,Allocator>& t, int){
            for(int i = 0; i < t.size(); i++)
                *this & t[i];
        }
        template<class T, class C, class Allocator> void load_override(boost::ptr_vector<T,C,Allocator>& t, int){
            for(int i = 0; i < t.size(); i++)
                *this & (t.c_array()[i]);
        }
        template<class T> void load_override(T*& t, int){
            *this & *t;
        }
        template<class T> void load_override(boost::serialization::nvp<T>& t, int){
            boost::archive::load(*this, t);
        }
        template<class T> void load(T& t){ 
            assert(false);
        }
        void load(int& t)          { /* ... */ }
        void load(unsigned long& t){ /* ... */ }
        // boost meta //
        void load_override(boost::archive::class_id_type& t, int){  }
        void load_override(boost::archive::object_id_type& t, int){  }
        void load_override(boost::archive::class_id_optional_type& t, int){  }
        void load_override(boost::archive::class_id_reference_type& t, int){  }
        void load_override(boost::archive::tracking_type& t, int){  }
        void load_override(boost::archive::version_type& t, int){  }
        void load_override(const boost::serialization::nvp<boost::serialization::collection_size_type>& t, int){  }
        void load_override(const boost::serialization::nvp<boost::serialization::item_version_type>& t, int){  }
    };

} }

#endif
