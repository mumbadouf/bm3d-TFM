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
,    unsigned width
,    unsigned height
);
 

//! Add boundaries by makeSymmetrical
void makeSymmetrical(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,    unsigned width
,    unsigned height
,    unsigned N
);
 
//! Look for the closest power of 2 number
int closest_power_of_2(
     unsigned n
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<unsigned> &ind_set
,    unsigned max_size
,    unsigned N
,    unsigned step
);
 
 


#endif // UTILITIES_H_INCLUDED
