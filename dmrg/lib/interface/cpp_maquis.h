/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University Departement of Chemistry
 *               2017-2018 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <map>
#include <memory>

namespace alps
{
    namespace numeric
    {
        template <class T, class MemoryBlock> class matrix;
    }
}

// Forward declaration for matrix, need to specifiy the Memory Block default argument
// Ideally, ALPS should provide a forward declaration header with the default template argument
typedef alps::numeric::matrix<double, std::vector<double> >                              matrix;
typedef alps::numeric::matrix<std::complex<double>, std::vector<std::complex<double> > > cmatrix;

template <class Matrix, class SymmGroup> class dmrg_sim;

class DmrgParameters;

class simulation_base {
public:
    virtual ~simulation_base() {}
    virtual void run() =0;
    virtual void measure_observable(std::string name,
                                    std::vector<double> & results, std::vector<std::vector<int> > & labels,
                                    std::string bra, std::shared_ptr<simulation_base> bra_ptr = NULL) =0;

    //virtual void add_ortho(std::shared_ptr<simulation_base> os) =0;
    virtual void add_ortho(simulation_base* os) =0;

    //virtual parameters::proxy get_parm(std::string const& key) =0;
};

template <class SymmGroup>
class simulation : public simulation_base {
public:
    simulation(DmrgParameters & parms);

    void run();

    void measure_observable(std::string name,
                            std::vector<double> & results, std::vector<std::vector<int> > & labels,
                            std::string bra,
                            std::shared_ptr<simulation_base> bra_ptr = NULL);

    //parameters::proxy get_parm(std::string const& key);

    //void add_ortho(std::shared_ptr<simulation_base> os);
    void add_ortho(simulation_base* os);

private:
    std::shared_ptr<dmrg_sim<matrix, SymmGroup> > sim_ptr_real;
    std::shared_ptr<dmrg_sim<cmatrix, SymmGroup> > sim_ptr_complex;
};

struct simulation_traits {
    typedef std::shared_ptr<simulation_base> shared_ptr;
    template <class SymmGroup> struct F {
        typedef simulation<SymmGroup> type;
    };
};


class Interface
{
public:
    Interface();
    Interface(std::map<std::string, std::string> & parms);

    //void SetParameters(DmrgParameters & p) { /*parms = p;*/ }

    //std::string value(std::string key);

    void optimize();
    void excite();

    void measure(std::string name, int bra, int ket);

    std::vector<double> getObservable(std::string name);

    std::vector<std::vector<int> > getLabels(std::string name);

    std::vector<double> opdm(int bra=0, int ket=0);
    std::vector<double> tpdm(int bra=0, int ket=0);
    void opdm(double **Gij, int bra=0, int ket=0);
    void tpdm(double **Gijkl, int bra=0, int ket=0);

private:
    std::map<std::string, std::vector<double> > observables;
    std::map<std::string, std::vector<std::vector<int> > > labels;

    std::vector<simulation_traits::shared_ptr> simv;
};

void prepare_integrals(double **, double **, double, int, int, std::map<std::string, std::string> &);
