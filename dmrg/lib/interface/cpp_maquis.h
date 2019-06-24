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
#include <stdexcept>
#include <utility>

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

class FrontEndBase {
public:
    virtual ~FrontEndBase() {}
    virtual void run() =0;
    virtual void measure_all() =0;
    virtual void measure_observable(std::string name,
                                    std::vector<double> & results, std::vector<std::vector<int> > & labels,
                                    std::string bra, std::shared_ptr<FrontEndBase> bra_ptr = NULL) =0;

    virtual double get_energy() =0;

    //virtual void add_ortho(std::shared_ptr<FrontEndBase> os) =0;
    virtual void add_ortho(FrontEndBase* os) =0;

    //virtual parameters::proxy get_parm(std::string const& key) =0;
};

template <class Matrix, class SymmGroup>
class SimFrontEnd : public FrontEndBase {
public:
    SimFrontEnd(DmrgParameters & parms);

    void run();

    void measure_all();

    void measure_observable(std::string name,
                            std::vector<double> & results, std::vector<std::vector<int> > & labels,
                            std::string bra,
                            std::shared_ptr<FrontEndBase> bra_ptr = NULL);

    double get_energy();

    //parameters::proxy get_parm(std::string const& key);

    //void add_ortho(std::shared_ptr<FrontEndBase> os);
    void add_ortho(FrontEndBase* os);

private:
    std::shared_ptr<dmrg_sim<Matrix, SymmGroup>> sim_ptr;
};

struct simulation_traits {
    typedef std::shared_ptr<FrontEndBase> shared_ptr;
    template <class Matrix, class SymmGroup> struct F {
        typedef SimFrontEnd<Matrix, SymmGroup> type;
    };
};


class Interface
{
public:
    Interface();
    Interface(DmrgParameters & parms, int spin_);

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

    double energy(int state);

private:

    void set_threads();
    void restore_threads();

    std::map<std::string, std::vector<double> > observables;
    std::map<std::string, std::vector<std::vector<int> > > labels;

    std::vector<simulation_traits::shared_ptr> simv;

    int tc_num_threads;

    int spin;
};


class DmrgInterface
{
public:
    DmrgInterface();
    DmrgInterface(std::map<std::string, std::string> & parms, int, int, int, int, int, int, int);

    void calc_states();

    void measure(std::string name, int bra, int ket);
    std::vector<double> getObservable(std::string name);
    std::vector<std::vector<int> > getLabels(std::string name);

    void opdm(double **Gij, int bra=0, int ket=0);
    void tpdm(double **Gijkl, int bra=0, int ket=0);

    double energy(int state);

private:
    std::pair<int, int> state_to_s_n(int state);

    constexpr static int max_spin = 7;

    Interface iface_[max_spin];
    int nstates[max_spin];
};

void prepare_integrals(double **, double **, double, int, int, std::map<std::string, std::string> &);
