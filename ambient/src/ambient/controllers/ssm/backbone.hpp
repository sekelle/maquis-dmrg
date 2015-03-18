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

#ifndef AMBIENT_CONTROLLERS_SSM_BACKBONE_HPP
#define AMBIENT_CONTROLLERS_SSM_BACKBONE_HPP

namespace ambient { 

        inline backbone::~backbone(){
            delete this->base_actor;
        }
        inline backbone::backbone() : sid(1) {
            this->base_actor = new actor_auto(provide_controller());
            context::init(base_actor);
            this->tag_ub = get_controller().get_tag_ub();
            this->num_procs = get_controller().get_num_procs();
            this->push_scope(new ambient::scope(num_procs));
            if(!get_controller().verbose()) this->io_guard.enable();
            if(ambient::isset("AMBIENT_VERBOSE")) this->info();
        }
        inline void backbone::info(){
            std::cout << "ambient: initialized ("                   << AMBIENT_THREADING_TAGLINE      << ")\n";
            std::cout << "ambient: size of instr bulk chunks: "     << AMBIENT_INSTR_BULK_CHUNK       << "\n";
            std::cout << "ambient: size of data bulk chunks: "      << AMBIENT_DATA_BULK_CHUNK        << "\n";
            if(ambient::isset("AMBIENT_BULK_LIMIT")) std::cout << "ambient: max share of data bulk: " << ambient::getint("AMBIENT_BULK_LIMIT") << "%\n";
            if(ambient::isset("AMBIENT_BULK_REUSE")) std::cout << "ambient: enabled bulk garbage collection\n";
            if(ambient::isset("AMBIENT_BULK_FORCE_FREE")) std::cout << "ambient: enabled bulk deallocation\n";
            #ifdef MPI_VERSION
            std::cout << "ambient: maximum tag value: "             << tag_ub                         << "\n";
            std::cout << "ambient: number of procs: "               << num_procs                      << "\n";
            #endif
            std::cout << "ambient: number of threads: "             << ambient::num_threads()         << "\n";
            std::cout << "\n";
        }
        inline int backbone::generate_sid(){
            return (++sid %= tag_ub);
        }
        inline int backbone::get_sid(){
            return sid;
        }
        inline int backbone::get_num_procs(){
            return num_procs;
        }
        inline typename backbone::controller_type& backbone::get_controller(){
            return *get_actor().controller; // caution: != context::get().controller
        }
        inline void backbone::revoke_controller(controller_type* c){
        }
        inline bool backbone::has_nested_actor(){
            return (&get_actor() != this->base_actor);
        }
        inline typename backbone::controller_type* backbone::provide_controller(){
            return &context::get().controller;
        }
        inline void backbone::sync(){
            context::sync();
            memory::data_bulk::drop();
        }
        inline actor& backbone::get_actor(){
            return *context::get().actors.top();
        }
        inline actor_auto& backbone::get_base_actor(){
            return *this->base_actor;
        }
        inline void backbone::pop_actor(){
            context::get().actors.pop();
        }
        inline void backbone::push_actor(actor* s){
            context::get().actors.push(s);
        }
        inline scope& backbone::get_scope(){
            return *context::get().scopes.top();
        }
        inline void backbone::pop_scope(){
            context::get().scopes.pop();
        }
        inline void backbone::push_scope(scope* s){
            context::get().scopes.push(s);
        }
        inline bool backbone::tunable(){
            return (!get_controller().is_serial() && !has_nested_actor());
        }
        inline void backbone::intend_read(models::ssm::revision* r){
            base_actor->intend_read(r); 
        }
        inline void backbone::intend_write(models::ssm::revision* r){
            base_actor->intend_write(r); 
        }
        inline void backbone::schedule(){
            base_actor->schedule();
        }
        inline ambient::mutex& backbone::get_mutex(){
            return mtx;
        }
}

#endif
