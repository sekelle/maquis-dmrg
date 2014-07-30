/*
 * Ambient Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
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

#ifndef AMBIENT_INTERFACE_TYPED
#define AMBIENT_INTERFACE_TYPED

#define EXTRACT(var) T* var = (T*)m->arguments[arg];

namespace ambient { namespace numeric {
    template <class T, class Allocator> class matrix;
    template <class T> class transpose_view;
    template <class T> class diagonal_matrix;
} }

namespace ambient {
    template<typename T> class default_allocator;
    using ambient::controllers::ssm::functor;
    using ambient::models::ssm::history;
    using ambient::models::ssm::revision;
    using ambient::memory::instr_bulk;

    // {{{ compile-time type info: singular types + inplace and future specializations
    template <typename T> struct singular_info {
        template<size_t arg> static void deallocate     (functor* m){                        }
        template<size_t arg> static bool pin            (functor* m){ return false;          }
        template<size_t arg> static void score          (T& obj)    {                        }
        template<size_t arg> static bool ready          (functor* m){ return true;           }
        template<size_t arg> static T&   revised        (functor* m){ EXTRACT(o); return *o; }
        template<size_t arg> static void modify (T& obj, functor* m){
            m->arguments[arg] = (void*)new(ambient::pool::malloc<instr_bulk,T>()) T(obj); 
        }
        template<size_t arg> static void modify_remote(T& obj)      {                        }
        template<size_t arg> static void modify_local(T& obj, functor* m){
            m->arguments[arg] = (void*)new(ambient::pool::malloc<instr_bulk,T>()) T(obj);
        }
    };
    template <typename T> struct singular_inplace_info : public singular_info<T> {
        template<size_t arg> static T& revised(functor* m){ return *(T*)&m->arguments[arg]; }
        template<size_t arg> static void modify_remote(T& obj){ }
        template<size_t arg> static void modify_local(T& obj, functor* m){ *(T*)&m->arguments[arg] = obj; }
        template<size_t arg> static void modify(T& obj, functor* m){ *(T*)&m->arguments[arg] = obj; }
    };
    template <typename T> struct future_info : public singular_info<T> {
        template<size_t arg> static void deallocate(functor* m){       
            EXTRACT(o); o->desc->generator = NULL;
        }
        template<size_t arg> static void modify_remote(T& obj){ 
            selector.get_controller().rsync(obj.desc);
        }
        template<size_t arg> static void modify_local(const T& obj, functor* m){ 
            obj.desc->generator = m;
            selector.get_controller().lsync(obj.desc);
            m->arguments[arg] = (void*)new(ambient::pool::malloc<instr_bulk,T>()) T(obj.desc);
        }
        template<size_t arg> static void modify(const T& obj, functor* m){ 
            m->arguments[arg] = (void*)new(ambient::pool::malloc<instr_bulk,T>()) T(obj.desc);
        }
    };
    template <typename T> struct read_future_info : public future_info<T> {
        template<size_t arg> static void deallocate(functor* m){ }
        template<size_t arg> static void modify_remote(T& obj){ }
        template<size_t arg> static void modify_local(const T& obj, functor* m){
            m->arguments[arg] = (void*)new(ambient::pool::malloc<instr_bulk,T>()) T(obj.desc);
        }
    };
    // }}}
    // {{{ compile-time type info: iteratable derived types
    template <typename T> struct iteratable_info : public singular_info<T> {
        template<size_t arg> 
        static void deallocate(functor* m){
            EXTRACT(o);
            revision& parent  = *o->ambient_before;
            revision& current = *o->ambient_after;
            current.complete();
            current.release();
            selector.get_controller().squeeze(&parent);
            parent.release();
        }
        template<size_t arg>
        static void modify_remote(T& obj){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            if(o->back()->owner != ambient::which())
                selector.get_controller().rsync(o->back());
            selector.get_controller().collect(o->back());
            selector.get_controller().add_revision<ambient::locality::remote>(o, ambient::which()); 
        }
        template<size_t arg>
        static void modify_local(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); 
            m->arguments[arg] = (void*)var;
            selector.get_controller().lsync(o->back());
            selector.get_controller().use_revision(o);
            selector.get_controller().collect(o->back());

            var->ambient_before = o->current;
            selector.get_controller().add_revision<ambient::locality::local>(o, m); 
            selector.get_controller().use_revision(o);
            var->ambient_after = o->current;
        }
        template<size_t arg>
        static void modify(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); m->arguments[arg] = (void*)var;
            selector.get_controller().sync(o->back());
            selector.get_controller().use_revision(o);
            selector.get_controller().collect(o->back());

            var->ambient_before = o->current;
            selector.get_controller().add_revision<ambient::locality::common>(o, m); 
            selector.get_controller().use_revision(o);
            var->ambient_after = o->current;
        }
        template<size_t arg>
        static T& revised(functor* m){ 
            EXTRACT(o); revise(*o);
            return *o;
        }
        template<size_t arg> 
        static bool pin(functor* m){ 
            EXTRACT(o);
            revision& r = *o->ambient_before;
            if(r.generator != NULL){
                ((functor*)r.generator)->queue(m);
                return true;
            }
            return false;
        }
        template<size_t arg> 
        static void score(T& obj){
            selector.intend_read(obj.ambient_rc.desc->back());
            selector.intend_write(obj.ambient_rc.desc->back());
        }
        template<size_t arg> 
        static bool ready(functor* m){
            EXTRACT(o);
            revision& r = *o->ambient_before;
            if(r.generator == NULL || r.generator == m) return true;
            return false;
        }
    };
    // }}}
    // {{{ compile-time type info: only read/write iteratable derived types

    template <typename T> struct read_iteratable_info : public iteratable_info<T> {
        template<size_t arg> static void deallocate(functor* m){
            EXTRACT(o);
            revision& r = *o->ambient_before;
            selector.get_controller().squeeze(&r);
            r.release();
        }
        template<size_t arg> static void modify_remote(T& obj){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            if(o->back()->owner != ambient::which())
                selector.get_controller().rsync(o->back());
        }
        template<size_t arg> static void modify_local(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); m->arguments[arg] = (void*)var;
            var->ambient_before = var->ambient_after = o->current;
            selector.get_controller().lsync(o->back());
            selector.get_controller().use_revision(o);
        }
        template<size_t arg> static void modify(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); m->arguments[arg] = (void*)var;
            var->ambient_before = var->ambient_after = o->current;
            selector.get_controller().sync(o->back());
            selector.get_controller().use_revision(o);
        }
        template<size_t arg> 
        static void score(T& obj){
            selector.intend_read(obj.ambient_rc.desc->back());
        }
        template<size_t arg> 
        static bool pin(functor* m){ 
            EXTRACT(o);
            revision& r = *o->ambient_before;
            if(r.generator != NULL){
                ambient::guard<ambient::mutex> g(selector.get_mutex());
                ((functor*)r.generator)->queue(m);
                return true;
            }
            return false;
        }
    };
    template <typename T> struct write_iteratable_info : public iteratable_info<T> {
        template<size_t arg> static void modify_remote(T& obj){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            selector.get_controller().collect(o->back());
            selector.get_controller().add_revision<ambient::locality::remote>(o, ambient::which()); 
        }
        template<size_t arg> static void modify_local(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); m->arguments[arg] = (void*)var;

            selector.get_controller().use_revision(o);
            selector.get_controller().collect(o->back());

            var->ambient_before = o->current;
            selector.get_controller().add_revision<ambient::locality::local>(o, m); 
            selector.get_controller().use_revision(o);
            var->ambient_after = o->current;
        }
        template<size_t arg> static void modify(T& obj, functor* m){
            decltype(obj.ambient_rc.desc) o = obj.ambient_rc.desc;
            selector.get_controller().touch(o);
            T* var = (T*)ambient::pool::malloc<instr_bulk,T>(); memcpy((void*)var, &obj, sizeof(T)); m->arguments[arg] = (void*)var;
            selector.get_controller().use_revision(o);
            selector.get_controller().collect(o->back());

            var->ambient_before = o->current;
            selector.get_controller().add_revision<ambient::locality::common>(o, m); 
            selector.get_controller().use_revision(o);
            var->ambient_after = o->current;
        }
        template<size_t arg> static bool pin(functor* m){ return false; }
        template<size_t arg> static void score(T& obj) {               
            selector.intend_write(obj.ambient_rc.desc->back());
        }
        template<size_t arg> static bool ready (functor* m){ return true;  }
    };
    // }}}

    // {{{ compile-time type info: specialization for forwarded types
    using ambient::numeric::future;
    using ambient::numeric::matrix;
    using ambient::numeric::diagonal_matrix;
    using ambient::numeric::transpose_view;

    template <typename T>
    struct has_allocator {
        template <typename T1> static typename T1::allocator_type test(int);
        template <typename>    static void test(...);
        enum { value = !std::is_void<decltype(test<T>(0))>::value };
    };
    template <bool HAS, typename T> struct checked_get_allocator {};
    template <typename T> struct checked_get_allocator<true, T> { typedef typename T::allocator_type type; };
    template <typename T> struct checked_get_allocator<false, T> { typedef typename ambient::default_allocator<T> type; }; // or T::value_type
    template <typename T> struct get_allocator { typedef typename checked_get_allocator<has_allocator<T>::value, T>::type type; };

    template <class T> struct unbound : public T {
        typedef typename get_allocator<T>::type allocator_type;
    };

    template<typename T> struct has_versioning {
        template<std::size_t V> struct valuekeeper {};
        template<typename R, typename C> static char helper(R(C::*)());
        template<typename C> static char check(valuekeeper<sizeof(helper(&C::ambient_enable_versioning))>*);
        template<typename C> static double check(...);
        enum { value = (sizeof(char) == sizeof(check<T>(0))) };
    };

    template <bool V, typename T> struct versioned_info { };
    template<typename T> struct versioned_info<true, T> { typedef iteratable_info< T > type; };
    template<typename T> struct versioned_info<false, T> { typedef singular_info< T > type; };

    template <bool V, typename T> struct const_versioned_info { };
    template<typename T> struct const_versioned_info<true, T> { typedef read_iteratable_info< const T > type; };
    template<typename T> struct const_versioned_info<false, T> { typedef singular_info< const T > type; };

    template <typename T>
    struct info {
        typedef typename       versioned_info<has_versioning<T>::value,T>::type typed;
        template <typename U> static U& unfold(T& naked){ return *static_cast<U*>(&naked); }
    };
    template <typename T>
    struct info <const T> {
        typedef typename const_versioned_info<has_versioning<T>::value,T>::type typed;
        template <typename U> static const T& unfold(const T& naked){ return naked; }
    };

    template <>
    struct info < size_t > {
        typedef size_t type;
        typedef singular_inplace_info<type> typed; 
        template <typename U> static type& unfold(type& naked){ return naked; }
    };

    template <typename S>
    struct info < future<S> > {
        typedef future<S> type;
        typedef future_info<type> typed; 
        template <typename U> static type& unfold(type& folded){ return folded.unfold(); }
    };

    template <typename S>
    struct info < const future<S> > { 
        typedef const future<S> type;
        typedef read_future_info<type> typed; 
        template <typename U> static type& unfold(type& folded){ return folded.unfold(); }
    };

    template <typename S>
    struct info < diagonal_matrix<S> > {
        typedef diagonal_matrix<S> type;
        template <typename U> static U& unfold(type& folded){ return *static_cast<U*>(&folded.get_data()); }
    };

    template <typename S>
    struct info < const diagonal_matrix<S> > {
        typedef const diagonal_matrix<S> type;
        template <typename U> static const matrix<S, ambient::default_allocator<S> >& unfold(type& folded){ return folded.get_data(); }
    };

    template <class Matrix>
    struct info < const transpose_view<Matrix> > {
        typedef const transpose_view<Matrix> type;
        template <typename U> static const Matrix& unfold(type& folded){ return *(const Matrix*)&folded; }
    };

    template <class Matrix>
    struct info < transpose_view<Matrix> > {
        typedef transpose_view<Matrix> type;
        template <typename U> static Matrix& unfold(type& folded){ return *(Matrix*)&folded; }
    };

    template <typename T>
    struct info < unbound<T> > {
        typedef unbound<T> type;
        typedef write_iteratable_info< type > typed; 
        template <typename U> static type& unfold(type& naked){ return naked; }
    };

    // }}}

    #define AMBIENT_DISABLE_DESTRUCTOR  static int  ambient_disable_destructor(int);
    #define AMBIENT_DELEGATE(...)       struct      ambient_type_structure;                                                                   \
                                        void        ambient_enable_versioning();                                                               \
                                        mutable     ambient::revision* ambient_before;                                                          \
                                        mutable     ambient::revision* ambient_after;                                                            \
                                        static void ambient_disable_destructor(...);                                                              \
                                        enum { ambient_destructor_disabled = !std::is_void<decltype(ambient_disable_destructor(0))>::value };      \
                                        struct ambient_desc {                                                                                       \
                                            typedef ambient::history rc_type;                                                                        \
                                            typedef ambient_type_structure mapping;                                                                   \
                                            ambient_desc(size_t n = sizeof(mapping)){ desc = new rc_type(ambient::dim2(1,1), n); }                     \
                                            ambient_desc(size_t n, size_t ts){ desc = new rc_type(ambient::dim2(1,n),ts); }                             \
                                            ambient_desc(size_t m, size_t n, size_t ts){ desc = new rc_type(ambient::dim2(n,m),ts); }                    \
                                            template<typename U> ambient_desc(const U& other):desc(other.desc){ }                                         \
                                            ambient_desc(rc_type* desc):desc(desc){ }                                                                      \
                                           ~ambient_desc(){ if(!ambient_destructor_disabled){ if(desc->weak()) delete desc; else ambient::destroy(desc); }} \
                                            rc_type* desc;                                                                                                   \
                                        } ambient_rc;                                                                                                         \
                                        struct ambient_type_structure { __VA_ARGS__ };

    #define AMBIENT_ALLOC(N, TYPE_SIZE)       ambient_rc(N, TYPE_SIZE)
    #define AMBIENT_ALLOC_2D(M, N, TYPE_SIZE) ambient_rc(M, N, TYPE_SIZE)
    #define AMBIENT_VAR_LENGTH 1
}

#undef EXTRACT
#endif
