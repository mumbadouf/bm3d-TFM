#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>
#include <omp.h>

#include "bm3d.h"
#include "utilities.h"

using namespace std;

// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
const char *pick_option(int *c, char **v, const char *o, const char *d) {
    int id = d ? 1 : 0;
    for (int i = 0; i < *c - id; i++) {
        if (v[i][0] == '-' && 0 == strcmp(v[i] + 1, o)) {
            char *r = v[i + id] + 1 - id;
            for (int j = i; j < *c - id; j++)
                v[j] = v[j + id + 1];
            *c -= id + 1;
            return r;
        }
    }
    return d;
}

/**
 * @file   main.cpp
 * @brief  Main executable file. Do not use lib_fftw to
 *         process DCT.
 *
 * @author MARC LEBRUN  <marc.lebrun@cmla.ens-cachan.fr>
 */
int main(int argc, char **argv) {
    //! Variables initialization
    const bool verbose = pick_option(&argc, argv, "verbose", nullptr) != nullptr;

    const int patch_size = 8;

    //! Check if there is the right call for the algorithm
    if (argc < 4) {
        cerr << "usage: " << argv[0] << " input sigma output [-verbose]" << endl;
        return EXIT_FAILURE;
    }

    //! Declarations
    vector<float> img_noisy, img_basic, img_denoised;
    unsigned width, height;

    //! Load image
    if (load_image(argv[1], img_noisy, &width, &height) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    float sigma = strtof(argv[2], nullptr);
    double start = omp_get_wtime();
    /*------------------------------------------------------------*/
    //! Denoising

    //! Parameters
    const unsigned nHard = 16; //! Half size of the search window
    const unsigned nWien = 16; //! Half size of the search window
    const unsigned pHard = 3;
    const unsigned pWien = 3;

    //! Overrides size if patch_size>0, else default behavior (8 or 12 depending on test)
    const unsigned kHard = patch_size;
    const unsigned kWien = patch_size;

    //! Check memory allocation
    img_basic.resize(img_noisy.size());
    img_denoised.resize(img_noisy.size());

    //! Add boundaries and make them Symmetrical
    const unsigned heightBoundary = height + 2 * nHard;
    const unsigned widthBoundary = width + 2 * nHard;
    vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;

    makeSymmetrical(img_noisy, img_sym_noisy, width, height, nHard);

    //! Denoising, 1st Step
    if (verbose)
        cout << "BM3D 1st step...";
    bm3d_1st_step(sigma, img_sym_noisy, img_sym_basic, widthBoundary, heightBoundary, nHard, kHard, pHard);
    if (verbose)
        cout << "is done." << endl;

    //! To avoid boundaries problem

    //copy img_sym_basic center (without boundaries) then make boundaries symmetrical
    unsigned dc_b = nHard * widthBoundary + nHard;
    unsigned dc = 0;
    for (unsigned i = 0; i < height; i++)
        for (unsigned j = 0; j < width; j++, dc++)
            img_basic[dc] = img_sym_basic[dc_b + i * widthBoundary + j];

    makeSymmetrical(img_basic, img_sym_basic, width, height, nHard);

    //! Denoising, 2nd Step
    if (verbose)
        cout << "BM3D 2nd step...";
    bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised, widthBoundary, heightBoundary, nWien, kWien,
                  pWien);
    if (verbose)
        cout << "is done." << endl;

    //! Obtention of img_denoised
    //copy img_sym_denoised center (without boundaries)
    dc_b = nWien * widthBoundary + nWien;
    dc = 0;
    for (unsigned i = 0; i < height; i++)
        for (unsigned j = 0; j < width; j++, dc++)
            img_denoised[dc] = img_sym_denoised[dc_b + i * widthBoundary + j];
    /*------------------------------------------------------------*/
    double end = omp_get_wtime();
    cout << "Time: " << end - start << "s" << endl;

    //! save noisy, denoised and differences images
    cout << endl << "Save images...";

    if (save_image(argv[3], img_denoised, width, height) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    cout << "done." << endl;

    return EXIT_SUCCESS;
}
