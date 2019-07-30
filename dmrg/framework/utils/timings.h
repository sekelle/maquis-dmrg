#ifndef MAQUIS_TIMINGS_H
#define MAQUIS_TIMINGS_H

#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include "utils/io.hpp"

class Timer
{
public:
    Timer(std::string name_)
    : val(0.0), name(name_), nCounter(0) { }
    
    ~Timer() { maquis::cout << name << " " << val << ", nCounter : " << nCounter << std::endl; }
    
    Timer & operator+=(double t) {
        val += t;
        return *this;
    }
    
    void begin() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    
    void end() {
		nCounter += 1;
        std::chrono::duration<double> sec = std::chrono::high_resolution_clock::now() - t0;
        val += sec.count();
    }
    
    double get_time() const {
	    return  val;
    }
    
    friend std::ostream& operator<< (std::ostream& os, Timer const& timer) {
        os << timer.name << " " << timer.val << ", nCounter : " << timer.nCounter;
        return os;
    }
    
protected:
    double val;
    std::string name;
    std::chrono::high_resolution_clock::time_point t0; 
    unsigned long long nCounter;
};

#endif
