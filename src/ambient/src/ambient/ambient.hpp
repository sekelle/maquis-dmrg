#ifndef AMBIENT
#define AMBIENT
// {{{ system includes
#include <mpi.h>
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <limits>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <memory.h>
#include <stdarg.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <pthread.h>
// }}}

#define AMBIENT_NUM_PROCS 2

#ifdef AMBIENT_CILK
    #include <cilk/cilk.h>
    #define AMBIENT_NUM_THREADS __cilkrts_get_total_workers()
    #define AMBIENT_THREAD_ID __cilkrts_get_worker_number()
    #define AMBIENT_THREAD cilk_spawn
    #define AMBIENT_SMP_ENABLE
    #define AMBIENT_SMP_DISABLE
#elif defined(AMBIENT_OMP)
    #include <omp.h>
    #define AMBIENT_THREAD_ID omp_get_thread_num()
    #define AMBIENT_PRAGMA(a) _Pragma( #a )
    #define AMBIENT_THREAD AMBIENT_PRAGMA(omp task untied)
    #define AMBIENT_SMP_ENABLE AMBIENT_PRAGMA(omp parallel) { AMBIENT_PRAGMA(omp single)
    #define AMBIENT_SMP_DISABLE }
    #define AMBIENT_NUM_THREADS [&]()->int{ int n; AMBIENT_SMP_ENABLE \
                                { n = omp_get_num_threads(); } \
                                AMBIENT_SMP_DISABLE return n; }()
#else
    #define AMBIENT_NUM_THREADS 1
    #define AMBIENT_THREAD_ID 0
    #define AMBIENT_THREAD
    #define AMBIENT_SMP_ENABLE
    #define AMBIENT_SMP_DISABLE
#endif

#ifdef AMBIENT_CPP11
    #define AMBIENT_MOVE(var) std::move(var)
#else
    #define AMBIENT_MOVE(var) var
#endif

namespace ambient {
    inline int get_num_threads(){
        static int n = AMBIENT_NUM_THREADS; return n;
    }
    enum complexity { N, N2, N3 };
    enum locality   { remote, local, common };
    enum scope_t    { base, single, shared };
}

#include "ambient/utils/memory.hpp"
#include "ambient/models/velvet/model.h"
#include "ambient/channels/mpi/channel.h"
#include "ambient/controllers/velvet/controller.h"
#include "ambient/utils/auxiliary.hpp"
#include "ambient/utils/io.hpp"
#include "ambient/interface/typed.hpp"
#include "ambient/interface/kernel.hpp"
#endif
