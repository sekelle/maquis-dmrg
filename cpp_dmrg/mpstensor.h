#ifndef MPSTENSOR_H
#define MPSTENSOR_H

#include "block_matrix.h"
#include "indexing.h"

#include <iostream>

enum MPSStorageLayout { LeftPaired, RightPaired };
// these are actually used in several places
enum Indicator { U, L, R };
enum DecompMethod { QR, SVD };

template<class Matrix, class SymmGroup>
class MPSTensor
{
public:
    typedef typename Matrix::value_type scalar_type;
    typedef double real_type;
    
    MPSTensor(Index<SymmGroup> const & sd = Index<SymmGroup>(),
              Index<SymmGroup> const & ld = Index<SymmGroup>(),
              Index<SymmGroup> const & rd = Index<SymmGroup>());
    
    Index<SymmGroup> site_dim() const;
    Index<SymmGroup> row_dim() const;
    Index<SymmGroup> col_dim() const;
    bool isobccompatible(Indicator) const;
    
    // these are not const because after a numerical test
    // they may update the status
    bool isleftnormalized(bool test = false);
    bool isrightnormalized(bool test = false);
    bool isnormalized(bool test = false);
    
    block_matrix<Matrix, SymmGroup> normalize_left(DecompMethod method = QR,
                                                   bool multiplied = true,
                                                   double truncation = 0,
                                                   Index<SymmGroup> bond_dim = Index<SymmGroup>());
    block_matrix<Matrix, SymmGroup> normalize_right(DecompMethod method = QR,
                                                    bool multiplied = true,
                                                    double truncation = 0,
                                                    Index<SymmGroup> bond_dim = Index<SymmGroup>());
    
    void multiply_from_left(block_matrix<Matrix, SymmGroup> &);
    void multiply_from_right(block_matrix<Matrix, SymmGroup> &);
    void multiply_by_scalar(scalar_type);
    
    scalar_type scalar_overlap(MPSTensor&);
    real_type scalar_norm();
    
    // this is completely useless in C++, only exists for consistency with Python
    MPSTensor copy() const;
    
    template<class Matrix_, class SymmGroup_>
    friend std::ostream& operator<<(std::ostream&, MPSTensor<Matrix_, SymmGroup_> const &);
    
    template<class Matrix_, class SymmGroup_>
    friend block_matrix<Matrix_, SymmGroup_> overlap_left_step(MPSTensor<Matrix_, SymmGroup_> & bra_tensor,
                                                               MPSTensor<Matrix_, SymmGroup_> & ket_tensor,
                                                               block_matrix<Matrix_, SymmGroup_> & left,
                                                               block_matrix<Matrix_, SymmGroup_> * local_op = NULL);
    
    template<class Matrix_, class SymmGroup_>
    friend block_matrix<Matrix_, SymmGroup_> overlap_right_step(MPSTensor<Matrix_, SymmGroup_> & bra_tensor,
                                                                MPSTensor<Matrix_, SymmGroup_> & ket_tensor,
                                                                block_matrix<Matrix_, SymmGroup_> & right,
                                                                block_matrix<Matrix_, SymmGroup_> * local_op = NULL);
    
private:
    Index<SymmGroup> phys_i, left_i, right_i;
    block_matrix<Matrix, SymmGroup> data_;
    MPSStorageLayout cur_storage;
    Indicator cur_normalization;
    
    void reflect();
    
    void make_left_paired();
    void make_right_paired();
};

template<class Matrix, class SymmGroup>
block_matrix<Matrix, SymmGroup> overlap_left_step(MPSTensor<Matrix, SymmGroup> & bratensor,
                                                  MPSTensor<Matrix, SymmGroup> & kqettensor,
                                                  block_matrix<Matrix, SymmGroup> & left,
                                                  block_matrix<Matrix, SymmGroup> * localop = NULL);

template<class Matrix, class SymmGroup>
block_matrix<Matrix, SymmGroup> overlap_right_step(MPSTensor<Matrix, SymmGroup> & bratensor,
                                                   MPSTensor<Matrix, SymmGroup> & kqettensor,
                                                   block_matrix<Matrix, SymmGroup> & right,
                                                   block_matrix<Matrix, SymmGroup> * localop = NULL);

#include "mpstensor.hpp"

#endif
