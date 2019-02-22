/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Department of Chemistry and the PULSE Institute, Stanford University
 *                    Laboratory for Physical Chemistry, ETH Zurich
 *               2017-2017 by Sebastian Keller <sebkelle@phys.ethz.ch>
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

#ifndef TOOLS_MATRIX_GROUP_HPP
#define TOOLS_MATRIX_GROUP_HPP

#include <vector>
#include <map>
#include <utility>

namespace contraction {
namespace common {

    using boost::get; 

    template <class Matrix, class SymmGroup>
    class MatrixGroupPrint
    {
        typedef typename MPOTensor<Matrix, SymmGroup>::index_type index_type;
        typedef typename Matrix::value_type value_type;

    public:

        typedef typename detail::micro_task<value_type> micro_task;

        MatrixGroupPrint() : valid(false) {}

        void add_line(unsigned b1, unsigned k)
        {
            // if size is zero or we see a new b1 for the first time and the previous b1 did yield terms
            if (bs.size() == 0 || (*bs.rbegin() != b1 && tasks.rbegin()->size() > 0))
            {
                bs.push_back(b1);
                ks.push_back(k);
                tasks.push_back(std::vector<micro_task>());
            }
            // if the previous b1 didnt yield any terms overwrite it with the new b1
            else if (*bs.rbegin() != b1 && tasks.rbegin()->size() == 0)
            {
                *bs.rbegin() = b1;
                *ks.rbegin() = k;
            }
        }

        void push_back(micro_task mt)
        {
            assert(tasks.size() > 0);
            tasks[tasks.size()-1].push_back(mt);
            valid = true;
        }

        std::vector<micro_task> & current_row()
        {
            return tasks[tasks.size()-1];
        }

        std::size_t n_tasks() const
        {
            std::size_t ret = 0;
            for (int i = 0; i < tasks.size(); ++i)
                ret += tasks[i].size();
            return ret;
        }

        void print_stats(MPOTensor<Matrix, SymmGroup> const & mpo) const
        {
            typedef boost::tuple<unsigned, unsigned, unsigned, unsigned> quadruple;
            typedef std::map<quadruple, unsigned> amap_t;

            int sw = 4;

            unsigned cnt = 0;
            amap_t b2_col;
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    quadruple tt(tasks[i][j].b2, tasks[i][j].k, tasks[i][j].in_offset, r_size); 
                    if (b2_col.count(tt) == 0)
                        b2_col[tt] = cnt++;
                }

            alps::numeric::matrix<double> alpha(tasks.size(), b2_col.size(), 0);
            for (int i = 0; i < tasks.size(); ++i)
                for (int j = 0; j < tasks[i].size(); ++j)
                {
                    quadruple tt(tasks[i][j].b2, tasks[i][j].k, tasks[i][j].in_offset, r_size); 
                    double val = tasks[i][j].scale;
                    alpha(i, b2_col[tt]) = (std::abs(val) > 1e-300) ? val : 1e-301;
                }

            int lpc = sw + 2 + sw;
            std::string leftpad(lpc, ' ');

            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << mpo.right_spin(boost::get<0>(it->first)).get();
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << mpo.num_col_non_zeros(boost::get<0>(it->first));
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<2>(it->first);
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<1>(it->first);
            maquis::cout << std::endl;
            maquis::cout << leftpad;
            for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                maquis::cout << std::setw(sw) << boost::get<0>(it->first);
            maquis::cout << std::endl;

            std::string hline(lpc + sw * b2_col.size(), '_');
            maquis::cout << hline << std::endl;

            for (int i = 0; i < bs.size(); ++i)
            {
                maquis::cout << std::setw(sw) << bs[i] << std::setw(sw) << ks[i] << "| ";
                for (amap_t::const_iterator it = b2_col.begin(); it != b2_col.end(); ++it)
                {
                    int col = it->second;
                    double val = alpha(i, col);
                    if (val == 0.)
                        maquis::cout << std::setw(sw) << "."; 
                    else
                        maquis::cout << std::setw(sw) << f3(alpha(i, col)); 
                }
                maquis::cout << std::endl;
            }
            maquis::cout << std::endl << std::endl;
        }

    //private:
        std::vector<std::vector<micro_task> > tasks;
        std::vector<index_type> bs, ks;

        bool valid;
        unsigned l_size, m_size, r_size, offset;
    private:
    };

} // namespace common
} // namespace contraction

#endif
