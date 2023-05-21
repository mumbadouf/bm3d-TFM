#ifndef LIB_TRANSFORMS_INCLUDED
#define LIB_TRANSFORMS_INCLUDED

#include<vector>

//! Compute a Bior1.5 2D
void bior_2d_forward(
    std::vector<float> const& input
,   std::vector<float> &output
,    unsigned N
,    unsigned d_i
,    unsigned r_i
,    unsigned d_o
,   std::vector<float> const& lpd
,   std::vector<float> const& hpd
);

//! Compute a Bior1.5 2D inverse
void bior_2d_inverse(
    std::vector<float> &signal
,    unsigned N
,    unsigned d_s
,   std::vector<float> const& lpr
,   std::vector<float> const& hpr
);

//! Precompute the Bior1.5 coefficients
void bior15_coef(
    std::vector<float> &low_freq_forward
,   std::vector<float> &high_freq_forward
,   std::vector<float> &low_freq_backward
,   std::vector<float> &high_freq_backward
);

//! Apply Walsh-Hadamard transform (non normalized) on a vector of size N = 2^n
void hadamard_transform(
    std::vector<float> &vec
,   std::vector<float> &tmp
,    unsigned N
,    unsigned d
);

//! Process the log2 of N
unsigned log2(
     unsigned N
);

//! Obtain index for periodic extension
void per_ext_ind(
    std::vector<unsigned> &ind_per
,    unsigned N
,    unsigned L
);

#endif // LIB_TRANSFORMS_INCLUDED
