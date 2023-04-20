#ifndef BM3D_H_INCLUDED
#define BM3D_H_INCLUDED
#include <vector>

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_bm3d(
    const float sigma
,   std::vector<float> &img_noisy
,   std::vector<float> &img_basic
,   std::vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned patch_size = 0
,   const bool verbose = false
);

//! 1st step of BM3D
void bm3d_1st_step(
    const float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> &img_basic
,   const unsigned width
,   const unsigned height
,   const unsigned nHard
,   const unsigned kHard
,   const unsigned NHard
,   const unsigned pHard
);

//! 2nd step of BM3D
void bm3d_2nd_step(
    const float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> const& img_basic
,   std::vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned nWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
);

//! Process 2D bior1.5 transform of a group of patches
void bior_2d_process(
    std::vector<float> &bior_table_2D
,   std::vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   const unsigned i_min
,   const unsigned i_max
,   std::vector<float> &lpd
,   std::vector<float> &hpd
);

void bior_2d_inverse(
    std::vector<float> &group_3D_table
,   const unsigned kHW
,   std::vector<float> const& lpr
,   std::vector<float> const& hpr
);

//! HT filtering using Welsh-Hadamard transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_hadamard(
    std::vector<float> &group_3D
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kHard
,   const float  sigma
,   const float lambdaThr3D
,   float * weight
);

//! Wiener filtering using Welsh-Hadamard transform
void wiener_filtering_hadamard(
    std::vector<float> &group_3D_img
,   std::vector<float> &group_3D_est
,   std::vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kWien
,   const float  sigma
,   float * weight
);

//! Apply a bior1.5 spline wavelet on a vector of size N x N.
void bior1_5_transform(
    std::vector<float> const& input
,   std::vector<float> &output
,   const unsigned N
,   std::vector<float> const& bior_table
,   const unsigned d_i
,   const unsigned d_o
,   const unsigned N_i
,   const unsigned N_o
);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
//! Preprocess coefficients of the Kaiser window and normalization coef for the DCT
void preProcess(
    std::vector<float> &kaiserWindow
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,   const unsigned kHW
);

void precompute_BM(
    std::vector<std::vector<unsigned> > &patch_table
,   const std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned n
,   const unsigned pHW
,   const float    tauMatch
);

#endif // BM3D_H_INCLUDED
