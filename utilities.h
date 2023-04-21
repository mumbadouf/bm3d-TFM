#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>

//! Read image and check number of channels
int load_image(
    char* name
,   std::vector<float> &img
,   unsigned * width
,   unsigned * height
);

//! Write image
int save_image(
    char* name
,   std::vector<float> &img
,   const unsigned width
,   const unsigned height
);
 

//! Add boundaries by symetry
void symetrize(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,   const unsigned width
,   const unsigned height
,   const unsigned N
);
 
//! Look for the closest power of 2 number
int closest_power_of_2(
    const unsigned n
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<unsigned> &ind_set
,   const unsigned max_size
,   const unsigned N
,   const unsigned step
);
 
 


#endif // UTILITIES_H_INCLUDED
