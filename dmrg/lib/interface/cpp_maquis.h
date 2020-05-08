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

#ifndef CPP_MAQUIS_H
#define CPP_MAQUIS_H

#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <tuple>



class DmrgParameters;
class FrontEndBase;

class State
{
public:

    State();
    State(const std::map<std::string, std::string>&);

    void optimize();

    State excite() const;

    std::string getParm(const std::string& key);

    std::map<std::string, std::string> getParameters();

    std::tuple<std::vector<double>, std::vector<int>> measure(std::string const& name) const;

private:
    State(DmrgParameters);

    int excitation=0;

    std::shared_ptr<FrontEndBase> simv;
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

    std::vector<std::shared_ptr<FrontEndBase>> simv;

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

#endif
