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
#include <cstdlib>

#include "utilities.h"

extern "C" {
#include "iio.h"
}
using namespace std;

/**
 * @brief Load image, check the number of channels
 *
 * @param name : name of the image to read
 * @param img : vector which will contain the image : R, G and B concatenated
 * @param width, height : size of the image
 *
 * @return EXIT_SUCCESS if the image has been loaded, EXIT_FAILURE otherwise
 **/
int load_image(char *name, vector<float> &img, unsigned *width, unsigned *height) {
    //! read input image
    cout << endl << "Read input image...";
    size_t h, w, c;
    float *tmp;
    int ih, iw, ic;

    tmp = iio_read_image_float_split(name, &iw, &ih, &ic);
    w = iw;
    h = ih;
    c = ic;
    if (!tmp) {
        cout << "error :: " << name << " not found or not a correct image" << endl;
        return EXIT_FAILURE;
    }
    cout << "done." << endl;

    //! test if image is really a color image and exclude the alpha channel
    if (c > 2) {
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
 * @param width, height : size of the image
 *
 * @return EXIT_SUCCESS if the image has been saved, EXIT_FAILURE otherwise
 **/
int save_image(char *name, std::vector<float> &img, const unsigned width, const unsigned height) {
    //! Allocate Memory
    auto *tmp = new float[width * height];

    //! Check for boundary problems
    for (unsigned k = 0; k < width * height; k++)
        tmp[k] = img[k]; //(img[k] > 255.0f ? 255.0f : (img[k] < 0.0f ? 0.0f : img[k]));

    iio_save_image_float_split(name, tmp, width, height, 1);

    //! Free Memory
    delete[] tmp;

    return EXIT_SUCCESS;
}


/**
 * @brief Add boundaries by symetry
 *
 * @param img : image to makeSymmetrical
 * @param img_sym : will contain img with symetrized boundaries
 * @param width, height : size of img
 * @param N : size of the boundary
 *
 * @return none.
 **/
void
makeSymmetrical(const std::vector<float> &img, std::vector<float> &img_sym, const unsigned width, const unsigned height,
                const unsigned N) {
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
        for (unsigned i = 0; i < N; i++) {
            img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w];
            img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w];
        }

    //! Right and left
    dc_2 = 0;
    for (unsigned i = 0; i < h; i++) {
        const unsigned di = dc_2 + i * w;
        for (unsigned j = 0; j < N; j++) {
            img_sym[di + j] = img_sym[di + 2 * N - j - 1];
            img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j];
        }
    }

}


/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(const unsigned n) {
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
void ind_initialize(vector<unsigned> &ind_set, const unsigned max_size, const unsigned N, const unsigned step) {
    ind_set.clear();
    unsigned ind = N;
    while (ind < max_size - N) {
        ind_set.push_back(ind);
        ind += step;
    }
    if (ind_set.back() < max_size - N - 1)
        ind_set.push_back(max_size - N - 1);
}
 
 
 
