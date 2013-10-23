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

#ifndef AMBIENT_NUMERIC_UTILS
#define AMBIENT_NUMERIC_UTILS
    
#ifdef AMBIENT_TRACKING
#include <ambient/utils/overseer.hpp>
#endif

namespace ambient {

    template<class Matrix>
    inline void make_persistent(const Matrix& a){
        ambient::make_persistent(a.core);
    }

    template<class Matrix>
    inline void make_persistent(const numeric::tiles<Matrix>& a){
        int size = a.data.size();
        for(int i = 0; i < size; i++){
            make_persistent(a[i]);
        }
    }

    template<class Matrix>
    inline void touch(const numeric::tiles<Matrix>& a){
        int size = a.data.size();
        for(int i = 0; i < size; i++){
            touch(a[i]);
        }
    }

    template<class Matrix>
    inline void migrate(const numeric::tiles<Matrix>& a){
        numeric::tiles<Matrix>& m = const_cast<numeric::tiles<Matrix>&>(a);
        int size = m.data.size();
        for(int i = 0; i < size; i++){
            migrate(m[i]);
        }
    }

#ifdef AMBIENT_TRACKING
    template<class Matrix>
    inline void track(const Matrix& a, const std::string& label){
        ambient::overseer::track(a.core, label);
    }

    template<typename T>
    inline void track(const diagonal_matrix<T>& a, const std::string& label){
        ambient::overseer::track(a.get_data().core, label);
    }

    template<class Matrix>
    inline void track(numeric::tiles<Matrix>& a, const std::string& label){
        int size = a.data.size();
        for(int i = 0; i < size; i++){
            track(a[i], label);
        }
    }
#endif

}

#endif
