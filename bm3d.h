#ifndef BM3D_H_INCLUDED
#define BM3D_H_INCLUDED
#include <vector>

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_bm3d(
    float sigma
,   std::vector<float> &img_noisy
,   std::vector<float> &img_basic
,   std::vector<float> &img_denoised
,    unsigned width
,    unsigned height
,    unsigned patch_size = 0
,    bool verbose = false
);

//! 1st step of BM3D
void bm3d_1st_step(
     float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> &img_basic
,    unsigned width
,    unsigned height
,    unsigned nHard
,    unsigned kHard
,    unsigned NHard
,    unsigned pHard
);

//! 2nd step of BM3D
void bm3d_2nd_step(
     float sigma
,   std::vector<float> const& img_noisy
,   std::vector<float> const& img_basic
,   std::vector<float> &img_denoised
,    unsigned width
,    unsigned height
,    unsigned nWien
,    unsigned kWien
,    unsigned NWien
,    unsigned pWien
);

//! Process 2D bior1.5 transform of a group of patches
void bior_2d_process(
    std::vector<float> &bior_table_2D
,   std::vector<float> const &img
,    unsigned nHW
,    unsigned width
,    unsigned kHW
,    unsigned i_r
,    unsigned step
,    unsigned i_min
,    unsigned i_max
,   std::vector<float> &lpd
,   std::vector<float> &hpd
);

void bior_2d_inverse(
    std::vector<float> &group_3D_table
,    unsigned kHW
,   std::vector<float> const &lpr
,   std::vector<float> const &hpr
);

//! HT filtering using Welsh-Hadamard transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_hadamard(
    std::vector<float> &group_3D
,   std::vector<float> &tmp
,    unsigned nSx_r
,    unsigned kHard
,    float  sigma
,    float lambdaThr3D
,   float * weight
);

//! Wiener filtering using Welsh-Hadamard transform
void wiener_filtering_hadamard(
    std::vector<float> &group_3D_img
,   std::vector<float> &group_3D_est
,   std::vector<float> &tmp
,    unsigned nSx_r
,    unsigned kWien
,    float  sigma
,   float * weight
);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
//! Preprocess coefficients of the Kaiser window and normalization coef for the DCT
void preProcess(
    std::vector<float> &kaiserWindow
,   std::vector<float> &coef_norm
,   std::vector<float> &coef_norm_inv
,    unsigned kHW
);

void precompute_BM(
    std::vector<std::vector<unsigned> > &patch_table
,   const std::vector<float> &img
,    unsigned width
,    unsigned height
,    unsigned kHW
,    unsigned NHW
,    unsigned n
,    unsigned pHW
,    float    tauMatch
);

#endif // BM3D_H_INCLUDED
