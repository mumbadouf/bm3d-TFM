#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>

#include "bm3d.h"
#include "utilities.h"
#include "lib_transforms.h"
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
        patch_size = 2;
        kHard = patch_size;// must be a power of 2
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
    vector<float> my_table_2D((2 * nHard + 1) * widthBoundary * kHard * kHard, 0.0f);//(row_ind.size(), vector<float>((2 * nHard + 1) * width * kHard_squared, 0.0f));
    unsigned local_rows = (heightBoundary - 2 * nHard) / ranks + nHard;
    if (my_rank == 0) {
        vector<unsigned> row_ind;
        vector<float> lpd, hpd, lpr, hpr;
        bior15_coef(lpd, hpd, lpr, hpr);
        ind_initialize(row_ind, heightBoundary - kHard + 1, nHard, pHard);

        vector<float> table_2D_temp((2 * nHard + 1) * widthBoundary * kHard*kHard, 0.0f);
        int rank = 0;
        unsigned rank_max = (rank + 1) * (local_rows - nHard) + nHard;
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++) {
            const unsigned i_row = row_ind[ind_i];
            //! Update of my_table_2D
            //each CPU depends on the calculation of the previous one
            if(rank < ranks-1){
                bior_2d_process(table_2D_temp, img_noisy, nHard, width, kHard, i_row, pHard, row_ind[0], row_ind.back(),
                                lpd, hpd);
                if (row_ind[ind_i+1] >= rank_max) {
                    rank++;
                    MPI_Send(table_2D_temp.data(), int(table_2D_temp.size()), MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                    rank_max = (rank + 1) * (local_rows - nHard) + nHard;
                }
            }
        }

    } else {
        MPI_Recv(my_table_2D.data(), int(my_table_2D.size()), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    bm3d_1st_step(sigma, img_sym_noisy, img_sym_basic, widthBoundary, heightBoundary, nHard, kHard, pHard, my_table_2D,ranks,my_rank,local_rows);
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
