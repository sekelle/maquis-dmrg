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

#ifndef AMBIENT_CHANNELS_MPI_COLLECTIVE
#define AMBIENT_CHANNELS_MPI_COLLECTIVE

namespace ambient { namespace channels { namespace mpi {

    using ambient::models::ssm::revision;
    using ambient::models::ssm::transformable;

    template<typename T>
    class bcast {
        typedef ambient::bulk_allocator<int> allocator;
    public:
        void dispatch();
        bcast(T& o, int root) : object(o), root(root), self(0) {}
        T& object;
        std::vector<int,allocator> tags;
        int root;
        int self;
        int size;
        int* list;
        request impl; 
        fence guard;
    };

    template<class T> class collective {};

    template<>
    class collective<revision> : public bcast<revision>, public memory::use_bulk_new<collective<revision> > {
        typedef ambient::bulk_allocator<int> allocator;
    public:
        collective(revision& r, int root);
        void operator += (int rank);
        bool involved();
        bool test();
        std::vector<bool,allocator> states;
        std::vector<int,allocator> tree;
    };

    template<>
    class collective<transformable> : public bcast<transformable>, public memory::use_bulk_new<collective<transformable> > {
        typedef ambient::bulk_allocator<int> allocator;
    public:
        collective(transformable& v, int root);
        bool test();
    };

} } }

#endif
