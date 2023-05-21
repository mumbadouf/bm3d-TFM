#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>
#include <string>

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
,    unsigned nHard
);
 
//! Look for the closest power of 2 number
int closest_power_of_2(
     unsigned n
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<unsigned> &ind_set
,    unsigned max_size
,    unsigned nHard
,    unsigned pHard
);

void print_vector(std::string const &msg,
    std::vector<float> const &vec,
    int width,
    int height);
void print_vector(const std::string& msg,
        std::vector<unsigned> const &vec,
        int width,
        int height);
void print_vector(const std::string& msg,
                  std::vector<unsigned> const &vec,
                  int width,
                  int height,
                  int depth);


#endif // UTILITIES_H_INCLUDED
