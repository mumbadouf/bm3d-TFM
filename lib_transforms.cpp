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
 * @file lib_transforms.cpp
 * @brief 1D and 2D wavelet transforms
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include "lib_transforms.h"
#include <cmath>


using namespace std;

/**
 * @brief Compute a full 2D Bior 1.5 spline wavelet (normalized)
 *
 * @param input: vector on which the transform will be applied;
 * @param output: will contain the result;
 * @param N: size of the 2D patch (N x N) on which the 2D transform
 *           is applied. Must be a power of 2;
 * @param d_i: for convenience. Shift for input to access to the patch;
 * @param r_i: for convenience. input(i, j) = input[d_i + i * r_i + j];
 * @param d_o: for convenience. Shift for output;
 * @param lpd: low frequencies coefficients for the forward Bior 1.5;
 * @param hpd: high frequencies coefficients for the forward Bior 1.5.
 *
 * @return none.
 **/
void bior_2d_forward(
    vector<float> const& input
,   vector<float> &output
,   const unsigned N
,   const unsigned d_i
,   const unsigned r_i
,   const unsigned d_o
,   vector<float> const& lpd
,   vector<float> const& hpd
){
    //! Initializing output
    for (unsigned i = 0; i < N; i++)
        for (unsigned j = 0; j < N; j++)
            output[i * N + j + d_o] = input[i * r_i + j + d_i];

    const unsigned iter_max = log2(N);
    unsigned N_1 = N;
    unsigned N_2 = N / 2;
    const unsigned S_1 = lpd.size();
    const unsigned S_2 = S_1 / 2 - 1;

    for (unsigned iter = 0; iter < iter_max; iter++)
    {
        //! Periodic extension index initialization
        vector<float> tmp(N_1 + 2 * S_2);
        vector<unsigned> ind_per(N_1 + 2 * S_2);
        per_ext_ind(ind_per, N_1, S_2);

        //! Implementing row filtering
        for (unsigned i = 0; i < N_1; i++)
        {
            //! Periodic extension of the signal in row
            for (unsigned j = 0; j < tmp.size(); j++)
                tmp[j] = output[d_o + i * N + ind_per[j]];

            //! Low and High frequencies filtering
            for (unsigned j = 0; j < N_2; j++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += tmp[k + j * 2] * lpd[k];
                    v_h += tmp[k + j * 2] * hpd[k];
                }
                output[d_o + i * N + j] = v_l;
                output[d_o + i * N + j + N_2] = v_h;
            }
        }

        //! Implementing column filtering
        for (unsigned j = 0; j < N_1; j++)
        {
            //! Periodic extension of the signal in column
            for (unsigned i = 0; i < tmp.size(); i++)
                tmp[i] = output[d_o + j + ind_per[i] * N];

            //! Low and High frequencies filtering
            for (unsigned i = 0; i < N_2; i++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += tmp[k + i * 2] * lpd[k];
                    v_h += tmp[k + i * 2] * hpd[k];
                }
                output[d_o + j + i * N] = v_l;
                output[d_o + j + (i + N_2) * N] = v_h;
            }
        }

        //! Sizes update
        N_1 /= 2;
        N_2 /= 2;
    }
}

/**
 * @brief Compute a full 2D Bior 1.5 spline wavelet inverse (normalized)
 *
 * @param signal: vector on which the transform will be applied; It
 *                will contain the result at the end;
 * @param N: size of the 2D patch (N x N) on which the 2D transform
 *           is applied. Must be a power of 2;
 * @param d_s: for convenience. Shift for signal to access to the patch;
 * @param lpr: low frequencies coefficients for the inverse Bior 1.5;
 * @param hpr: high frequencies coefficients for the inverse Bior 1.5.
 *
 * @return none.
 **/
void bior_2d_inverse(
    vector<float> &signal
,   const unsigned N
,   const unsigned d_s
,   vector<float> const& lpr
,   vector<float> const& hpr
){
    //! Initialization
    const unsigned iter_max = log2(N);
    unsigned N_1 = 2;
    unsigned N_2 = 1;
    const unsigned S_1 = lpr.size();
    const unsigned S_2 = S_1 / 2 - 1;

    for (unsigned iter = 0; iter < iter_max; iter++)
    {

        vector<float> tmp(N_1 + S_2 * N_1);
        vector<unsigned> ind_per(N_1 + 2 * S_2 * N_2);
        per_ext_ind(ind_per, N_1, S_2 * N_2);

        //! Implementing column filtering
        for (unsigned j = 0; j < N_1; j++)
        {
            //! Periodic extension of the signal in column
            for (unsigned i = 0; i < tmp.size(); i++)
                tmp[i] = signal[d_s + j + ind_per[i] * N];

            //! Low and High frequencies filtering
            for (unsigned i = 0; i < N_2; i++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + i];
                    v_h += hpr[k] * tmp[k * N_2 + i];
                }

                signal[d_s + i * 2 * N + j] = v_h;
                signal[d_s + (i * 2 + 1) * N + j] = v_l;
            }
        }

        //! Implementing row filtering
        for (unsigned i = 0; i < N_1; i++)
        {
            //! Periodic extension of the signal in row
            for (unsigned j = 0; j < tmp.size(); j++)
                tmp[j] = signal[d_s + i * N + ind_per[j]];

            //! Low and High frequencies filtering
            for (unsigned j = 0; j < N_2; j++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for (unsigned k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + j];
                    v_h += hpr[k] * tmp[k * N_2 + j];
                }

                signal[d_s + i * N + j * 2] = v_h;
                signal[d_s + i * N + j * 2 + 1] = v_l;
            }
        }

        //! Sizes update
        N_1 *= 2;
        N_2 *= 2;
    }
}

/**
 * @brief Initialize forward and backward low and high filter
 *        for a Bior1.5 spline wavelet.
 *
 * @param low_freq_forward: low frequencies forward filter;
 * @param high_freq_forward: high frequencies forward filter;
 * @param low_freq_backward: low frequencies backward filter;
 * @param high_freq_backward: high frequencies backward filter.
 **/
void bior15_coef(
    vector<float> &low_freq_forward
,   vector<float> &high_freq_forward
,   vector<float> &low_freq_backward
,   vector<float> &high_freq_backward
){
    const float coef_norm = 1.f / (sqrtf(2.f) * 128.f);
    const float sqrt2_inv = 1.f / sqrtf(2.f);

    low_freq_forward.resize(10);
    low_freq_forward[0] =  3.f  ;
    low_freq_forward[1] = -3.f  ;
    low_freq_forward[2] = -22.f ;
    low_freq_forward[3] =  22.f ;
    low_freq_forward[4] =  128.f;
    low_freq_forward[5] =  128.f;
    low_freq_forward[6] =  22.f ;
    low_freq_forward[7] = -22.f ;
    low_freq_forward[8] = -3.f  ;
    low_freq_forward[9] =  3.f  ;

    high_freq_forward.resize(10);
    high_freq_forward[0] =  0.f;
    high_freq_forward[1] =  0.f;
    high_freq_forward[2] =  0.f;
    high_freq_forward[3] =  0.f;
    high_freq_forward[4] = -sqrt2_inv;
    high_freq_forward[5] =  sqrt2_inv;
    high_freq_forward[6] =  0.f;
    high_freq_forward[7] =  0.f;
    high_freq_forward[8] =  0.f;
    high_freq_forward[9] =  0.f;

    low_freq_backward.resize(10);
    low_freq_backward[0] = 0.f;
    low_freq_backward[1] = 0.f;
    low_freq_backward[2] = 0.f;
    low_freq_backward[3] = 0.f;
    low_freq_backward[4] = sqrt2_inv;
    low_freq_backward[5] = sqrt2_inv;
    low_freq_backward[6] = 0.f;
    low_freq_backward[7] = 0.f;
    low_freq_backward[8] = 0.f;
    low_freq_backward[9] = 0.f;

    high_freq_backward.resize(10);
    high_freq_backward[0] =  3.f  ;
    high_freq_backward[1] =  3.f  ;
    high_freq_backward[2] = -22.f ;
    high_freq_backward[3] = -22.f ;
    high_freq_backward[4] =  128.f;
    high_freq_backward[5] = -128.f;
    high_freq_backward[6] =  22.f ;
    high_freq_backward[7] =  22.f ;
    high_freq_backward[8] = -3.f  ;
    high_freq_backward[9] = -3.f  ;

    for (unsigned k = 0; k < 10; k++)
    {
        low_freq_forward[k] *= coef_norm;
        high_freq_backward[k] *= coef_norm;
    }
}

/**
 * @brief Apply Welsh-Hadamard transform on vec (non normalized !!)
 *
 * @param vec: vector on which a Hadamard transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param N, d: the Hadamard transform will be applied on vec[d] -> vec[d + N].
 *        N must be a power of 2!!!!
 *
 * @return None.
 **/
void hadamard_transform(
    vector<float> &vec
,   vector<float> &tmp
,   const unsigned N
,   const unsigned D
){
    if (N == 1)
        return;
    else if (N == 2)
    {
        const float a = vec[D + 0];
        const float b = vec[D + 1];
        vec[D + 0] = a + b;
        vec[D + 1] = a - b;
    }
    else
    {
        const unsigned n = N / 2;
        for (unsigned k = 0; k < n; k++)
        {
            const float a = vec[D + 2 * k];
            const float b = vec[D + 2 * k + 1];
            vec[D + k] = a + b;
            tmp[k] = a - b;
        }
        for (unsigned k = 0; k < n; k++)
            vec[D + n + k] = tmp[k];

        hadamard_transform(vec, tmp, n, D);
        hadamard_transform(vec, tmp, n, D + n);
    }
}

/**
 * @brief Obtain the ceil of log_2(N)
 *
 * @param N: in the case N = 2^n, return n.
 *
 * @return n;
 **/
unsigned log2(
    const unsigned N
){
    unsigned k = 1;
    unsigned n = 0;
    while (k < N)
    {
        k *= 2;
        n++;
    }
    return n;
}

/**
 * @brief Obtain index for periodic extension.
 *
 * @param ind_per: will contain index. Its size must be N + 2 * L;
 * @param N: size of the original signal;
 * @param L: size of boundaries to add on each side of the signal.
 *
 * @return none.
 **/
void per_ext_ind(
    vector<unsigned> &ind_per
,   const unsigned N
,   const unsigned L
){
    for (unsigned k = 0; k < N; k++)
        ind_per[k + L] = k;

    int ind1 = (N - L);
    while (ind1 < 0)
        ind1 += N;
    unsigned ind2 = 0;
    unsigned k = 0;
    while(k < L)
    {
        ind_per[k] = (unsigned) ind1;
        ind_per[k + L + N] = ind2;
        ind1 = ((unsigned) ind1 < N - 1 ? (unsigned) ind1 + 1 : 0);
        ind2 = (ind2 < N - 1 ? ind2 + 1 : 0);
        k++;
    }
}
