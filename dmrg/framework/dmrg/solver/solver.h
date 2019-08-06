
#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

#include "davidson_vector.h"
#include "tasks.h"

template <class T>
double solve(std::vector<std::size_t> const&, std::vector<const T*> const&, std::vector<T*> &,
             std::vector<const T*> const&, std::vector<const T*> const&, contraction::common::ScheduleNew<T>,
             std::vector<DavidsonVector<T>> const&,
             double, double, int);

#endif
