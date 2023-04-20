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
 * @file utilities.cpp
 * @brief Utilities functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "utilities.h"
extern "C"
{
#include "iio.h"
}

#define YUV 0
#define YCBCR 1
#define OPP 2
#define RGB 3

using namespace std;

/**
 * @brief Load image, check the number of channels
 *
 * @param name : name of the image to read
 * @param img : vector which will contain the image : R, G and B concatenated
 * @param width, height, chnls : size of the image
 *
 * @return EXIT_SUCCESS if the image has been loaded, EXIT_FAILURE otherwise
 **/
int load_image(
    char *name, vector<float> &img, unsigned *width, unsigned *height)
{
    //! read input image
    cout << endl
         << "Read input image...";
    size_t h, w, c;
    float *tmp = NULL;
    int ih, iw, ic;

    tmp = iio_read_image_float_split(name, &iw, &ih, &ic);
    w = iw;
    h = ih;
    c = ic;
    if (!tmp)
    {
        cout << "error :: " << name << " not found or not a correct image" << endl;
        return EXIT_FAILURE;
    }
    cout << "done." << endl;

    //! test if image is really a color image and exclude the alpha channel
    if (c > 2)
    {
        unsigned k = 0;
        while (k < w * h && tmp[k] == tmp[w * h + k] && tmp[k] == tmp[2 * w * h + k])
            k++;
        c = (k == w * h ? 1 : 3);
    }

    //! Some image informations
    cout << "image size :" << endl;
    cout << " - width          = " << w << endl;
    cout << " - height         = " << h << endl;
    cout << " - nb of channels = " << c << endl;

    //! Initializations
    *width = w;
    *height = h;

    img.resize(w * h);
    for (unsigned k = 0; k < w * h * c; k++)
        img[k] = tmp[k];

    return EXIT_SUCCESS;
}

/**
 * @brief write image
 *
 * @param name : path+name+extension of the image
 * @param img : vector which contains the image
 * @param width, height, chnls : size of the image
 *
 * @return EXIT_SUCCESS if the image has been saved, EXIT_FAILURE otherwise
 **/
int save_image(
    char *name, std::vector<float> &img, const unsigned width, const unsigned height)
{
    //! Allocate Memory
    float *tmp = new float[width * height];

    //! Check for boundary problems
    for (unsigned k = 0; k < width * height; k++)
        tmp[k] = img[k]; //(img[k] > 255.0f ? 255.0f : (img[k] < 0.0f ? 0.0f : img[k]));

    iio_save_image_float_split(name, tmp, width, height, 1);

    //! Free Memory
    delete[] tmp;

    return EXIT_SUCCESS;
}

/**
 * @brief Check if a number is a power of 2
 **/
bool power_of_2(
    const unsigned n)
{
    if (n == 0)
        return false;

    if (n == 1)
        return true;

    if (n % 2 == 0)
        return power_of_2(n / 2);
    else
        return false;
}

/**
 * @brief Add boundaries by symetry
 *
 * @param img : image to symetrize
 * @param img_sym : will contain img with symetrized boundaries
 * @param width, height, chnls : size of img
 * @param N : size of the boundary
 *
 * @return none.
 **/
void symetrize(
    const std::vector<float> &img, std::vector<float> &img_sym, const unsigned width, const unsigned height, const unsigned N)
{
    //! Declaration
    const unsigned w = width + 2 * N;
    const unsigned h = height + 2 * N;

    if (img_sym.size() != w * h)
        img_sym.resize(w * h);

    unsigned dc = 0;
    unsigned dc_2 = 0 + N * w + N;

    //! Center of the image
    for (unsigned i = 0; i < height; i++)
        for (unsigned j = 0; j < width; j++, dc++)
            img_sym[dc_2 + i * w + j] = img[dc];

    //! Top and bottom
    dc_2 = 0;
    for (unsigned j = 0; j < w; j++, dc_2++)
        for (unsigned i = 0; i < N; i++)
        {
            img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w];
            img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w];
        }

    //! Right and left
    dc_2 = 0;
    for (unsigned i = 0; i < h; i++)
    {
        const unsigned di = dc_2 + i * w;
        for (unsigned j = 0; j < N; j++)
        {
            img_sym[di + j] = img_sym[di + 2 * N - j - 1];
            img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j];
        }
    }

    return;
}

/**
 * @brief Compute PSNR and RMSE between img_1 and img_2
 *
 * @param img_1 : pointer to an allocated array of pixels.
 * @param img_2 : pointer to an allocated array of pixels.
 * @param psnr  : will contain the PSNR
 * @param rmse  : will contain the RMSE
 *
 * @return EXIT_FAILURE if both images haven't the same size.
 **/
int compute_psnr(
    const vector<float> &img_1, const vector<float> &img_2, float *psnr, float *rmse)
{
    if (img_1.size() != img_2.size())
    {
        cout << "Can't compute PSNR & RMSE: images have different sizes: " << endl;
        cout << "img_1 : " << img_1.size() << endl;
        cout << "img_2 : " << img_2.size() << endl;
        return EXIT_FAILURE;
    }

    float tmp = 0.0f;
    for (unsigned k = 0; k < img_1.size(); k++)
        tmp += (img_1[k] - img_2[k]) * (img_1[k] - img_2[k]);

    (*rmse) = sqrtf(tmp / (float)img_1.size());
    (*psnr) = 20.0f * log10f(255.0f / (*rmse));

    return EXIT_SUCCESS;
}

/**
 * @brief Compute a difference image between img_1 and img_2
 **/
int compute_diff(
    const std::vector<float> &img_1, const std::vector<float> &img_2, std::vector<float> &img_diff, const float sigma)
{
    if (img_1.size() != img_2.size())
    {
        cout << "Can't compute difference, img_1 and img_2 don't have the same size" << endl;
        cout << "img_1 : " << img_1.size() << endl;
        cout << "img_2 : " << img_2.size() << endl;
        return EXIT_FAILURE;
    }

    const unsigned size = img_1.size();

    if (img_diff.size() != size)
        img_diff.resize(size);

    const float s = 4.0f * sigma;

    for (unsigned k = 0; k < size; k++)
    {
        float value = (img_1[k] - img_2[k] + s) * 255.0f / (2.0f * s);
        img_diff[k] = (value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value));
    }

    return EXIT_SUCCESS;
}

/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(
    const unsigned n)
{
    unsigned r = 1;
    while (r * 2 <= n)
        r *= 2;

    return r;
}

/**
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param N : boundary;
 * @param step: step between two indices.
 *
 * @return none.
 **/
void ind_initialize(
    vector<unsigned> &ind_set, const unsigned max_size, const unsigned N, const unsigned step)
{
    ind_set.clear();
    unsigned ind = N;
    while (ind < max_size - N)
    {
        ind_set.push_back(ind);
        ind += step;
    }
    if (ind_set.back() < max_size - N - 1)
        ind_set.push_back(max_size - N - 1);
}

/**
 * @brief For convenience. Estimate the size of the ind_set vector built
 *        with the function ind_initialize().
 *
 * @return size of ind_set vector built in ind_initialize().
 **/
unsigned ind_size(
    const unsigned max_size, const unsigned N, const unsigned step)
{
    unsigned ind = N;
    unsigned k = 0;
    while (ind < max_size - N)
    {
        k++;
        ind += step;
    }
    if (ind - step < max_size - N - 1)
        k++;

    return k;
}

/**
 * @brief Initialize a 2D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the patch to apply the 2D transform;
 * @param kind: forward or backward;
 * @param nb: number of 2D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_2d(
    fftwf_plan *plan, const unsigned N, const fftwf_r2r_kind kind, const unsigned nb)
{
    int nb_table[2] = {N, N};
    int nembed[2] = {N, N};
    fftwf_r2r_kind kind_table[2] = {kind, kind};

    float *vec = (float *)fftwf_malloc(N * N * nb * sizeof(float));
    (*plan) = fftwf_plan_many_r2r(2, nb_table, nb, vec, nembed, 1, N * N, vec,
                                  nembed, 1, N * N, kind_table, FFTW_ESTIMATE);

    fftwf_free(vec);
}

/**
 * @brief Initialize a 1D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the vector to apply the 1D transform;
 * @param kind: forward or backward;
 * @param nb: number of 1D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_1d(
    fftwf_plan *plan, const unsigned N, const fftwf_r2r_kind kind, const unsigned nb)
{
    int nb_table[1] = {N};
    int nembed[1] = {N * nb};
    fftwf_r2r_kind kind_table[1] = {kind};

    float *vec = (float *)fftwf_malloc(N * nb * sizeof(float));
    (*plan) = fftwf_plan_many_r2r(1, nb_table, nb, vec, nembed, 1, N, vec,
                                  nembed, 1, N, kind_table, FFTW_ESTIMATE);
    fftwf_free(vec);
}

/**
 * @brief tabulated values of log2(N), where N = 2 ^ n.
 *
 * @param N : must be a power of 2 smaller than 64
 *
 * @return n = log2(N)
 **/
unsigned ind_log2(
    const unsigned N)
{
    return (N == 1 ? 0 : (N == 2 ? 1 : (N == 4 ? 2 : (N == 8 ? 3 : (N == 16 ? 4 : (N == 32 ? 5 : 6))))));
}

/**
 * @brief tabulated values of log2(N), where N = 2 ^ n.
 *
 * @param N : must be a power of 2 smaller than 64
 *
 * @return n = 2 ^ N
 **/
unsigned ind_pow2(
    const unsigned N)
{
    return (N == 0 ? 1 : (N == 1 ? 2 : (N == 2 ? 4 : (N == 3 ? 8 : (N == 4 ? 16 : (N == 5 ? 32 : 64))))));
}
