/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file bm3d.cpp
 * @brief BM3D de-noising functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <algorithm>
#include <cmath>
#include <iostream>

#include "bm3d.h"
#include "utilities.h"
#include "lib_transforms.h"

#define SQRT2 1.414213562373095
#define SQRT2_INV 0.7071067811865475

/*
 * In order to reproduce the original BM3D the DC coefficients are
 * thresholded (DCTHRESH uncommented) and are filtered using Wiener
 * (DCWIENER uncommented), MTRICK activates undocumented tricks from
 * Marc Lebrun's implementation of BM3D available in IPOL
 * http://www.ipol.im/pub/art/2012/l-bm3d/, not in the original paper.
 */

using namespace std;

bool compareFirst(pair<float, unsigned> pair1, pair<float, unsigned> pair2) {
    return pair1.first < pair2.first;
}

/**
 * @brief Run the basic process of BM3D (1st step). The result
 *        is contained in img_basic. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to de-noise;
 * @param img_noisy: noisy image;
 * @param img_basic: will contain the denoised image after the 1st step;
 * @param width, height : size of img_noisy;
 * @param nHard: size of the boundary around img_noisy;

 * @return none.
 **/
void bm3d_1st_step(const float sigma, vector<float> const &img_noisy, vector<float> &img_basic, const unsigned width,
                   const unsigned height, const unsigned nHard, const unsigned kHard, const unsigned pHard) {

    //! Parameters initialization
    const float lambdaHard3D = 2.7f;                              //! Threshold for Hard Thresholding
    const float tauMatch =
            (3.f) * (sigma < 35.0f ? 2500.f : 5000.f); //! threshold used to determinate similarity between patches
    const unsigned kHard_squared = kHard * kHard;
    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_squared);
    vector<float> coef_norm(kHard_squared);
    vector<float> coef_norm_inv(kHard_squared);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    //! Initialization for convenience
    vector<unsigned> row_ind;
    vector<unsigned> column_ind;
    vector<float> group_3D_table;
    vector<float> wx_r_table;
    vector<float> hadamard_tmp(nHard);

    //! For aggregation part
    vector<float> denominator(width * height, 0.0f);
    vector<float> numerator(width * height, 0.0f);

    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);

    group_3D_table.resize(kHard_squared * nHard * column_ind.size());
    wx_r_table.reserve(column_ind.size());

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());

    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);
    bior15_coef(lpd, hpd, lpr, hpr);
    //! Precompute Bloc-Matching
    vector<vector<unsigned>> patch_table;
    //  MPI STARTS HERE
    precompute_BM(patch_table, img_noisy, width, height, kHard, nHard, pHard, tauMatch);

    //! table_2D[p * N + q + (i * width + j) * kHard_squared + c * (2 * nHard + 1) * width * kHard_squared]
    vector<float> table_2D((2 * nHard + 1) * width * kHard_squared, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++) {
        const unsigned i_r = row_ind[ind_i];

        //! Update of table_2D
        bior_2d_process(table_2D, img_noisy, nHard, width, kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);

        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++) {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D(nSx_r * kHard_squared, 0.0f);

            for (unsigned n = 0; n < nSx_r; n++) {
                const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * width;
                for (unsigned k = 0; k < kHard_squared; k++)
                    group_3D[n + k * nSx_r + 0] = table_2D[k + ind * kHard_squared + 0];
            }

            //! HT filtering of the 3D group
            float weight;
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, sigma, lambdaHard3D, &weight);

            //! Save the 3D group. The DCT 2D inverse will be done after.

            for (unsigned n = 0; n < nSx_r; n++)
                for (unsigned k = 0; k < kHard_squared; k++)
                    group_3D_table.push_back(group_3D[n + k * nSx_r + 0]);

            //! Save weighting

            wx_r_table.push_back(weight);

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        bior_2d_inverse(group_3D_table, kHard, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++) {
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();

            for (unsigned n = 0; n < nSx_r; n++) {
                const unsigned k = patch_table[k_r][n] + 0;
                for (unsigned p = 0; p < kHard; p++)
                    for (unsigned q = 0; q < kHard; q++) {
                        const unsigned ind = k + p * width + q;
                        numerator[ind] += kaiser_window[p * kHard + q] * wx_r_table[ind_j] *
                                          group_3D_table[p * kHard + q + n * kHard_squared + 0 + dec];
                        denominator[ind] += kaiser_window[p * kHard + q] * wx_r_table[ind_j];
                    }
            }

            dec += nSx_r * kHard_squared;
        }

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height; k++)
        img_basic[k] = numerator[k] / denominator[k];
}

/**
 * @brief Run the final process of BM3D (2nd step). The result
 *        is contained in img_denoised. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: contains the denoised image after the 1st step;
 * @param img_denoised: will contain the final estimate of the denoised
 *        image after the second step;
 * @param width, height : size of img_noisy;
 * @param nWien: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the second step, otherwise use the norm
 *        of Wiener coefficients of the 3D group;

 *
 * @return none.
 **/
void bm3d_2nd_step(const float sigma, vector<float> const &img_noisy, vector<float> const &img_basic,
                   vector<float> &img_denoised, const unsigned width, const unsigned height, const unsigned nWien,
                   const unsigned kWien, const unsigned pWien) {
    //! Parameters initialization
    const float tauMatch = (sigma < 35.0f ? 400.f
                                          : 3500.f); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kWien + 1, nWien, pWien);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kWien + 1, nWien, pWien);
    const unsigned kWien_2 = kWien * kWien;
    vector<float> group_3D_table(kWien_2 * nWien * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(column_ind.size());
    vector<float> tmp(nWien);

    //! Check allocation memory
    if (img_denoised.size() != img_noisy.size())
        img_denoised.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kWien_2);
    vector<float> coef_norm(kWien_2);
    vector<float> coef_norm_inv(kWien_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kWien);

    //! For aggregation part
    vector<float> denominator(width * height, 0.0f);
    vector<float> numerator(width * height, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<unsigned>> patch_table;
    precompute_BM(patch_table, img_basic, width, height, kWien, nWien, pWien, tauMatch);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! DCT_table_2D[p * N + q + (i * width + j) * kWien_2 + c * (2 * ns + 1) * width * kWien_2]
    vector<float> table_2D_img((2 * nWien + 1) * width * kWien_2, 0.0f);
    vector<float> table_2D_est((2 * nWien + 1) * width * kWien_2, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++) {
        const unsigned i_r = row_ind[ind_i];

        //! Update of DCT_table_2D
        bior_2d_process(table_2D_img, img_noisy, nWien, width, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
        bior_2d_process(table_2D_est, img_basic, nWien, width, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);

        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++) {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D_est(nSx_r * kWien_2, 0.0f);
            vector<float> group_3D_img(nSx_r * kWien_2, 0.0f);

            for (unsigned n = 0; n < nSx_r; n++) {
                const unsigned ind = patch_table[k_r][n] + (nWien - i_r) * width;
                for (unsigned k = 0; k < kWien_2; k++) {
                    group_3D_est[n + k * nSx_r + 0] = table_2D_est[k + ind * kWien_2 + 0];
                    group_3D_img[n + k * nSx_r + 0] = table_2D_img[k + ind * kWien_2 + 0];
                }
            }

            //! Wiener filtering of the 3D group
            float weight;
            wiener_filtering_hadamard(group_3D_img, group_3D_est, tmp, nSx_r, kWien, sigma, &weight);

            //! Save the 3D group. The DCT 2D inverse will be done after.

            for (unsigned n = 0; n < nSx_r; n++)
                for (unsigned k = 0; k < kWien_2; k++)
                    group_3D_table.push_back(group_3D_est[n + k * nSx_r]);

            //! Save weighting

            wx_r_table.push_back(weight);

        } //! End of loop on j_r

        //!  Apply 2D bior inverse
        bior_2d_inverse(group_3D_table, kWien, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++) {
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();

            for (unsigned n = 0; n < nSx_r; n++) {
                const unsigned k = patch_table[k_r][n];
                for (unsigned p = 0; p < kWien; p++)
                    for (unsigned q = 0; q < kWien; q++) {
                        const unsigned ind = k + p * width + q;
                        numerator[ind] += kaiser_window[p * kWien + q] * wx_r_table[ind_j] *
                                          group_3D_table[p * kWien + q + n * kWien_2 + dec];
                        denominator[ind] += kaiser_window[p * kWien + q] * wx_r_table[ind_j];
                    }
            }

            dec += nSx_r * kWien_2;
        }

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height; k++)
        img_denoised[k] = numerator[k] / denominator[k];
}

/**
 * @brief Precompute a 2D bior1.5 transform on all patches contained in
 *        a part of the image.
 *
 * @param bior_table_2D : will contain the 2d bior1.5 transform for all
 *        chosen patches;
 * @param img : image on which the 2d transform will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height: size of img;
 * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it;
 * @param lpd : low pass filter of the forward bior1.5 2d transform;
 * @param hpd : high pass filter of the forward bior1.5 2d transform.
 **/
void bior_2d_process(vector<float> &bior_table_2D, vector<float> const &img, const unsigned nHW, const unsigned width,
                     const unsigned kHW, const unsigned i_r, const unsigned step, const unsigned i_min,
                     const unsigned i_max, vector<float> &lpd, vector<float> &hpd) {
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    //! If i_r == ns, then we have to process all Bior1.5 transforms
    if (i_r == i_min || i_r == i_max) {

        for (unsigned i = 0; i < 2 * nHW + 1; i++)
            for (unsigned j = 0; j < width - kHW; j++) {
                bior_2d_forward(img, bior_table_2D, kHW, (i_r + i - nHW) * width + j, width, (i * width + j) * kHW_2,
                                lpd, hpd);
            }
    } else {
        const unsigned ds = step * width * kHW_2;

        //! Re-use of Bior1.5 already processed

        for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
            for (unsigned j = 0; j < width - kHW; j++)
                for (unsigned k = 0; k < kHW_2; k++)
                    bior_table_2D[k + (i * width + j) * kHW_2] = bior_table_2D[k + (i * width + j) * kHW_2 + ds];

        //! Compute the new Bior

        for (unsigned i = 0; i < step; i++)
            for (unsigned j = 0; j < width - kHW; j++) {
                bior_2d_forward(img, bior_table_2D, kHW, (i + 2 * nHW + 1 - step + i_r - nHW) * width + j, width,
                                ((i + 2 * nHW + 1 - step) * width + j) * kHW_2, lpd, hpd);
            }
    }
}

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only third
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_3D : contains the 3D block for a reference patch;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kHW : size of patches (kHW x kHW);
 * @param sigma : contains value of noise for each channel;
 * @param lambdaHard3D : value of thresholding;
 * @param weight: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void ht_filtering_hadamard(vector<float> &group_3D, vector<float> &tmp, const unsigned nSx_r, const unsigned kHard,
                           const float sigma, const float lambdaHard3D, float *weight) {
    //! Declarations
    const unsigned kHard_2 = kHard * kHard;

    *weight = 0.0f;
    const float coef_norm = sqrtf((float) nSx_r);
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kHard_2; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding

    const float T = lambdaHard3D * sigma * coef_norm;
    for (unsigned k = 0; k < kHard_2 * nSx_r; k++) {

        if (fabs(group_3D[k]) > T)
            (*weight)++;
        else
            group_3D[k] = 0.0f;
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kHard_2; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    for (unsigned k = 0; k < group_3D.size(); k++)
        group_3D[k] *= coef;

    //! Weight for aggregation
    (*weight) = ((*weight) > 0.0f ? 1.0f / (float) (sigma * sigma * (*weight)) : 1.0f);
}

/**
 * @brief Wiener filtering using Hadamard transform.
 *
 * @param group_3D_img : contains the 3D block built on img_noisy;
 * @param group_3D_est : contains the 3D block built on img_basic;
 * @param tmp: allocated vector used in hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kWien : size of patches (kWien x kWien);
 * @param sigma : contains value of noise for each channel;
 * @param weight: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void wiener_filtering_hadamard(vector<float> &group_3D_img, vector<float> &group_3D_est, vector<float> &tmp,
                               const unsigned nSx_r, const unsigned kWien, const float sigma, float *weight) {
    //! Declarations
    const unsigned kWien_2 = kWien * kWien;
    const float coef = 1.0f / (float) nSx_r;

    (*weight) = 0.0f;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kWien_2; n++) {
        hadamard_transform(group_3D_img, tmp, nSx_r, n * nSx_r);
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);
    }

    //! Wiener Filtering

    for (unsigned k = 0; k < kWien_2 * nSx_r; k++) {
        float value = group_3D_est[k] * group_3D_est[k] * coef;
        value /= (value + sigma * sigma);
        group_3D_est[k] = group_3D_img[k] * value * coef;
        (*weight) += (value * value);
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kWien_2; n++)
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);

    //! Weight for aggregation
    (*weight) = ((*weight) > 0.0f ? 1.0f / (float) (sigma * sigma * (*weight)) : 1.0f);
}

void
bior_2d_inverse(vector<float> &group_3D_table, const unsigned kHW, vector<float> const &lpr, vector<float> const &hpr) {
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned N = group_3D_table.size() / kHW_2;

    //! Bior process
    for (unsigned n = 0; n < N; n++)
        bior_2d_inverse(group_3D_table, kHW, n * kHW_2, lpr, hpr);
}

/** ----------------- **/
/** - Preprocessing - **/
/** ----------------- **/
/**
 * @brief Preprocess
 *
 * @param kaiser_window[kHW * kHW]: Will contain values of a Kaiser Window;
 * @param coef_norm: Will contain values used to normalize the 2D DCT;
 * @param coef_norm_inv: Will contain values used to normalize the 2D DCT;
 * @param bior1_5_for: will contain coefficients for the bior1.5 forward transform
 * @param bior1_5_inv: will contain coefficients for the bior1.5 inverse transform
 * @param kHW: size of patches (need to be 8 or 12).
 *
 * @return none.
 **/
void
preProcess(vector<float> &kaiserWindow, vector<float> &coef_norm, vector<float> &coef_norm_inv, const unsigned kHW) {
    //! Kaiser Window coefficients

    //! First quarter of the matrix
    kaiserWindow[0 + kHW * 0] = 0.1924f;
    kaiserWindow[0 + kHW * 1] = 0.2989f;
    kaiserWindow[0 + kHW * 2] = 0.3846f;
    kaiserWindow[0 + kHW * 3] = 0.4325f;
    kaiserWindow[1 + kHW * 0] = 0.2989f;
    kaiserWindow[1 + kHW * 1] = 0.4642f;
    kaiserWindow[1 + kHW * 2] = 0.5974f;
    kaiserWindow[1 + kHW * 3] = 0.6717f;
    kaiserWindow[2 + kHW * 0] = 0.3846f;
    kaiserWindow[2 + kHW * 1] = 0.5974f;
    kaiserWindow[2 + kHW * 2] = 0.7688f;
    kaiserWindow[2 + kHW * 3] = 0.8644f;
    kaiserWindow[3 + kHW * 0] = 0.4325f;
    kaiserWindow[3 + kHW * 1] = 0.6717f;
    kaiserWindow[3 + kHW * 2] = 0.8644f;
    kaiserWindow[3 + kHW * 3] = 0.9718f;

    //! Completing the rest of the matrix by symmetry
    for (unsigned i = 0; i < kHW / 2; i++)
        for (unsigned j = kHW / 2; j < kHW; j++)
            kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

    for (unsigned i = kHW / 2; i < kHW; i++)
        for (unsigned j = 0; j < kHW; j++)
            kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];

    //! Coefficient of normalization for DCT II and DCT II inverse
    const float coef = 0.5f / ((float) (kHW));
    for (unsigned i = 0; i < kHW; i++)
        for (unsigned j = 0; j < kHW; j++) {
            if (i == 0 && j == 0) {
                coef_norm[0] = 0.5f * coef;
                coef_norm_inv[0] = 2.0f;
            } else if (i * j == 0) {
                coef_norm[i * kHW + j] = SQRT2_INV * coef;
                coef_norm_inv[i * kHW + j] = SQRT2;
            } else {
                coef_norm[i * kHW + j] = coef;
                coef_norm_inv[i * kHW + j] = 1.0f;
            }
        }
}

/**
 * @brief Precompute Block Matching (distance inter-patches)
 * applying the Haar filter to compute raw, noisy pixel distances (eq 4).
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordinate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param patch_size: size of patch
 * @param nHard: maximum similar patches wanted (half-size of search window)
 * @param pHard: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *
 * @return none.
 **/
void precompute_BM(vector<vector<unsigned>> &patch_table, const vector<float> &img, const unsigned width,
                   const unsigned height, const unsigned patch_size, const unsigned nHard, const unsigned pHard,
                   const float tauMatch) {
    //! Declarations
    //nHard= 16;pHard= 3;patch_size= 8;
    const unsigned Ns = 2 * nHard + 1;
    const float threshold = tauMatch * (float)(patch_size * patch_size);
    vector<float> diff_table(width * height);//same size as img
    vector<vector<float>> sum_table((nHard + 1) * Ns, vector<float>(width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    vector<unsigned> row_ind;
    vector<unsigned> column_ind;

    //! For each possible distance, precompute inter-patches distance
    for (unsigned i_nHard = 0; i_nHard <= nHard; i_nHard++) { //0 to 16
        for (unsigned dj = 0; dj < Ns; dj++) { //0 to 33(exclusive)
            const int dk = (int) (i_nHard * width + dj) - (int) nHard;
            const unsigned ddk = i_nHard * Ns + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned row = nHard; row < height - nHard; row++) {
                unsigned k = row * width + nHard;
                for (unsigned col = nHard; col < width - nHard; col++, k++) {
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
                    cout << k << " " << dk << endl;
                    cout <<  img[k] << " " << img[k + dk]  << endl;
                }
//                cout << "row loop"<<endl;
            }
//            cout << "NS Loop" << endl;

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHard * width + nHard;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < patch_size; p++) {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < patch_size; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHard; j < width - nHard; j++) {
                const unsigned ind = nHard * width + j;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < patch_size; p++) {
                    sum += diff_table[ind + p * width + patch_size] - diff_table[ind + p * width];
                }
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHard + 1; i < height - nHard; i++) {
                const unsigned ind = (i - 1) * width + nHard;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < patch_size; q++) {
                    sum += diff_table[ind + patch_size * width + q] - diff_table[ind + q];
                }
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHard + 1;
                unsigned pq = (i + patch_size - 1) * width + patch_size - 1 + nHard + 1;
                for (unsigned j = nHard + 1; j < width - nHard; j++, k++, pq++) {
                    sum_table[ddk][k] =
                            sum_table[ddk][k - 1] + sum_table[ddk][k - width] - sum_table[ddk][k - 1 - width] +
                            diff_table[pq] - diff_table[pq - patch_size] - diff_table[pq - patch_size * width] +
                            diff_table[pq - patch_size - patch_size * width];
                }
            }
        }
//        cout << "i_nHard loop" << endl;
    }
    ind_initialize(row_ind, height - patch_size + 1, nHard, pHard);
    ind_initialize(column_ind, width - patch_size + 1, nHard, pHard);
    //! Precompute Bloc Matching
    vector<pair<float, unsigned>> table_distance;
    //! To avoid reallocation
    table_distance.reserve(Ns * Ns);

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++) {
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++) {
            //! Initialization
            const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
            table_distance.clear();
            patch_table[k_r].clear();

            //! Threshold distances in order to keep similar patches
            for (int dj = -(int) nHard; dj <= (int) nHard; dj++) {
                for (int di = 0; di <= (int) nHard; di++)
                    if (sum_table[dj + nHard + di * Ns][k_r] < threshold)
                        table_distance.push_back(
                                make_pair(sum_table[dj + nHard + di * Ns][k_r], k_r + di * width + dj));

                for (int di = -(int) nHard; di < 0; di++)
                    if (sum_table[-dj + nHard + (-di) * Ns][k_r + di * width + dj] < threshold)
                        table_distance.push_back(make_pair(sum_table[-dj + nHard + (-di) * Ns][k_r + di * width + dj],
                                                           k_r + di * width + dj));
            }

            //! We need a power of 2 for the number of similar patches,
            //! because of the Welsh-Hadamard transform on the third dimension.
            //! We assume that nHard is already a power of 2
            const unsigned nSx_r = (nHard > table_distance.size() ? closest_power_of_2(table_distance.size()) : nHard);

            //! To avoid problem
            if (nSx_r == 1 && table_distance.empty()) {

                table_distance.push_back(make_pair(0, k_r));
            }

            //! Sort patches according to their distance to the reference one
            partial_sort(table_distance.begin(), table_distance.begin() + nSx_r, table_distance.end(), compareFirst);

            //! Keep a maximum of nHard similar patches
            for (unsigned n = 0; n < nSx_r; n++)
                patch_table[k_r].push_back(table_distance[n].second);

        }
    }
}
