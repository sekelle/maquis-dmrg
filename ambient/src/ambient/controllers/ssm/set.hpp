/*
 * Copyright Institute for Theoretical Physics, ETH Zurich 2015.
 * Distributed under the Boost Software License, Version 1.0.
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

namespace ambient { namespace controllers { namespace ssm {

    // {{{ transformable

    inline void set<transformable>::spawn(transformable& t){
        ((functor*)t.generator)->queue(new set(t));
    }
    inline set<transformable>::set(transformable& t) : t(t) {
        handle = ambient::select().get_controller().get_channel().bcast(t, ambient::which());
    }
    inline bool set<transformable>::ready(){
        return (t.generator != NULL ? false : handle->test());
    }
    inline void set<transformable>::invoke(){}

    // }}}
    // {{{ revision

    inline void set<revision>::spawn(revision& r){
        if(ambient::select().threaded()){ meta::spawn(r, meta::type::set); return; }
        set*& transfer = (set*&)r.assist.second;
        if(ambient::select().get_controller().update(r)) transfer = new set(r);
        *transfer += ambient::which();
        ambient::select().generate_sid();
    }
    inline set<revision>::set(revision& r) : t(r) {
        t.use();
        handle = ambient::select().get_controller().get_channel().set(t);
        if(t.generator != NULL) ((functor*)t.generator.load())->queue(this);
        else ambient::select().get_controller().queue(this);
    }
    inline void set<revision>::operator += (rank_t rank){
        *handle += rank;
    }
    inline bool set<revision>::ready(){
        return (t.generator != NULL ? false : handle->test());
    }
    inline void set<revision>::invoke(){
        ambient::select().get_controller().squeeze(&t);
        t.release(); 
    }

    // }}}

} } }
