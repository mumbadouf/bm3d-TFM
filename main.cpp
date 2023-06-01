#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>
#include <omp.h>

#include "bm3d.h"
#include "utilities.h"
#include "mpi.h"

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
    MPI_Init(&argc, &argv);
    int my_rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    //! Variables initialization
    const bool verbose = pick_option(&argc, argv, "verbose", nullptr) != nullptr;
    //! Parameters
    unsigned nHard = 16;//16; //! Half ranks of the search window
    unsigned nWien = 16;//16; //! Half ranks of the search window
    const unsigned pHard = 3;
    const unsigned pWien = 3;
    int patch_size = 8;
    unsigned kHard = patch_size;
    unsigned kWien = patch_size;
    double start, end;
    //! Check if there is the right call for the algorithm
    if (argc < 4) {
        cerr << "usage: " << argv[0] << " input sigma output [-verbose]" << endl;
        return EXIT_FAILURE;
    }
    float sigma = strtof(argv[2], nullptr);
    vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;
    //! Add boundaries and make them Symmetrical
    unsigned heightBoundary;
    unsigned widthBoundary;
    //! Declarations
    vector<float> img_noisy, img_basic, img_denoised;
    unsigned width, height;
    int testing = 1;
    //! Load image
    if (testing == 0) {
        width = 10;
        height = 10;
        nHard = 4;//16; //! Half ranks of the search window
        nWien = 4;//16; //! Half ranks of the search window
        patch_size = 3;
        kHard = patch_size;
        kWien = patch_size;
        img_noisy.resize(width * height);
        for (int i = 0; i < (int) height; i++)
            for (int j = 0; j < (int) width; j++)
                img_noisy[i * width + j] = (float) (i * width + j);
    } else {

        if (load_image(argv[1], img_noisy, &width, &height) != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    //  print_vector("img_noisy:",img_noisy, (int)width, (int)height);
    //start = MPI_Wtime();
    /*------------------------------------------------------------*/
    //! Denoising

    //! Check memory allocation
    img_basic.resize(img_noisy.size());
    img_denoised.resize(img_noisy.size());

    //! Add boundaries and make them Symmetrical
    heightBoundary = height + 2 * nHard;
    widthBoundary = width + 2 * nHard;


    makeSymmetrical(img_noisy, img_sym_noisy, width, height, nHard);
//    for (unsigned i = 0; i < heightBoundary; i++){
//        for (unsigned j = 0; j < widthBoundary; ++j) {
//            cout << i * widthBoundary + j<< " ";
//        }
//        cout << endl;
//    }
    //print_vector("img_sym_noisy",img_sym_noisy, (int)widthBoundary, (int)heightBoundary);
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
    if (my_rank == 0) {
        for (unsigned i = 0; i < height; i++)
            for (unsigned j = 0; j < width; j++, dc++)
                img_basic[dc] = img_sym_basic[dc_b + i * widthBoundary + j];

        makeSymmetrical(img_basic, img_sym_basic, width, height, nHard);

    }
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
   // end = MPI_Wtime();
   // cout << "Time: " << end - start << "s" << endl;

    //! save noisy, denoised and differences images
    cout << endl << "Save images...";
    char r[10];
    std::sprintf(r, "_%d.png", my_rank);
    //std::sprintf(r, "_%d.png", 0);

    if (save_image(strcat(argv[3], r), img_denoised, width, height) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    cout << "done." << endl;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
