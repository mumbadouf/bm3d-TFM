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
const char *pick_option(int *c, char **v, const char *o, const char *d)
{
    int id = d ? 1 : 0;
    for (int i = 0; i < *c - id; i++)
    {
        if (v[i][0] == '-' && 0 == strcmp(v[i] + 1, o))
        {
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
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int my_rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    //! Variables initialization
    const bool verbose = true; // pick_option(&argc, argv, "verbose", nullptr) != nullptr;
    //! Parameters
    unsigned nHard = 16; // 16; //! Half ranks of the search window
    unsigned nWien = 16; // 16; //! Half ranks of the search window
    const unsigned pHard = 3;
    const unsigned pWien = 3;
    int patch_size = 8;
    unsigned kHard = patch_size;
    unsigned kWien = patch_size;
    double start, end;
    //! Check if there is the right call for the algorithm
    if (argc < 4)
    {
        cerr << "usage: " << argv[0] << " input sigma output [-verbose]" << endl;
        return EXIT_FAILURE;
    }
    float sigma = strtof(argv[2], nullptr);
    vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;
    vector<float> my_img_sym_noisy, my_img_sym_basic, my_img_sym_denoised;
    //! Add boundaries and make them Symmetrical
    unsigned heightBoundary;
    unsigned widthBoundary;
    //! Declarations
    vector<float> img_noisy, img_basic, img_denoised;
    unsigned width, height;
    int testing = 1;
    //! Load image
    if (my_rank == 0)
    {
        if (testing == 0)
        {
            width = 1280;
            height = 800;
            nHard = 16; // 16; //! Half ranks of the search window
            nWien = 16; // 16; //! Half ranks of the search window
            patch_size = 8;
            kHard = patch_size;
            kWien = patch_size;
            img_noisy.resize(width * height);
            for (int i = 0; i < (int)height; i++)
                for (int j = 0; j < (int)width; j++)
                    img_noisy[i * width + j] = (float)(i * width + j);
        }
        else
        {

            if (load_image(argv[1], img_noisy, &width, &height) != EXIT_SUCCESS)
                return EXIT_FAILURE;
        }
        //  print_vector("img_noisy:",img_noisy, (int)width, (int)height);

        start = MPI_Wtime();
        /*------------------------------------------------------------*/
        //! Denoising

        //! Add boundaries and make them Symmetrical
        heightBoundary = height + 2 * nHard;
        widthBoundary = width + 2 * nHard;
        makeSymmetrical(img_noisy, img_sym_noisy, width, height, nHard);
    }
    else
    {
        if (testing == 0)
        {
            nHard = 16; // 16; //! Half ranks of the search window
            nWien = 16; // 16; //! Half ranks of the search window
            patch_size = 8;
            kHard = patch_size;
            kWien = patch_size;
        }
    }

    MPI_Bcast(&heightBoundary, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&widthBoundary, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int local_rows = int((heightBoundary - (2 * nHard)) / ranks);
    // cout << "Rank: " << my_rank << " " << heightBoundary << "x" << widthBoundary << " rows: " << local_rows << endl;
    vector<int> counts(ranks, int(local_rows * widthBoundary));
    vector<int> displs(ranks);

    displs[0] = nHard * widthBoundary;
    for (int rank = 1; rank < ranks; rank++)
    {
        displs[rank] = displs[rank - 1] + counts[rank - 1];
    }
    for (int rank = 0; rank < ranks; rank++)
    {
        counts[rank] += int(2 * nHard * widthBoundary);
        displs[rank] -= int(nHard * widthBoundary);
    }
    if (my_rank == ranks - 1)
    {
        local_rows += (heightBoundary - (2 * nHard)) % ranks;
        counts[ranks - 1] += ((heightBoundary - (2 * nHard)) % ranks) * widthBoundary;
    }
    cout << "Rank: " << my_rank << " Count: " << counts[my_rank] << " displs: " << displs[my_rank] << endl;
    my_img_sym_noisy.resize((local_rows + (2 * nHard)) * widthBoundary);
    // cout << "Rank: " << my_rank << " Before Scatter" << endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(&img_sym_noisy[0], &counts[0], &displs[0], MPI_FLOAT, &my_img_sym_noisy[0], counts[my_rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    // cout << "Rank: " << my_rank << " After Scatter" << endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // calculate local table_2D for Bior
    int my_table_2D_size = (2 * nHard + 1) * widthBoundary * kHard * kHard;
    vector<float> my_table_2D(my_table_2D_size, 0.0f);
    vector<unsigned> row_ind;
    vector<float> lpd, hpd, lpr, hpr;

    if (my_rank == 0)
    {
        bior15_coef(lpd, hpd, lpr, hpr);
        ind_initialize(row_ind, heightBoundary - kHard + 1, nHard, pHard);
        vector<float> table_2D_temp(my_table_2D_size, 0.0f);
        int rank = 0;
        unsigned rank_max = (rank + 1) * (local_rows) + nHard;
        for (unsigned ind_i = 0; rank < ranks - 1; ind_i++)
        {
            const unsigned i_row = row_ind[ind_i];
            //! Update of my_table_2D
            // each CPU depends on the calculation of the previous one

            bior_2d_process(table_2D_temp, img_sym_noisy, nHard, widthBoundary, kHard, i_row, pHard, row_ind[0],
                            row_ind.back(), lpd, hpd);
            if (row_ind[ind_i + 1] >= rank_max)
            {
                rank++;
                MPI_Send(&table_2D_temp[0], my_table_2D_size, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
                rank_max = (rank + 1) * (local_rows) + nHard;
            }
        }
        table_2D_temp.clear();
    }
    else
    {
        MPI_Recv(&my_table_2D[0], my_table_2D_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // cout << "Rank: " << my_rank << " After calculating my_table_2d" << endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    //! Denoising, 1st Step
    if (verbose && my_rank ==0)
        cout << "BM3D 1st step..."<<endl;
    bm3d_1st_step(sigma, my_img_sym_noisy, my_img_sym_basic, widthBoundary, heightBoundary, nHard, kHard, pHard, ranks,
                  my_rank, int(local_rows + (2 * nHard)), my_table_2D);

    cout << "Rank: " << my_rank << " finished step 1" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < ranks; i++)
    {
        counts[i] -= int(2 * nHard * widthBoundary);
    }
    for (int i = 0; i < ranks; i++)
    {
        displs[i] += int(nHard * widthBoundary);
    }


    if (my_rank == 0)
        img_sym_basic.resize(widthBoundary * heightBoundary);
    MPI_Gatherv(&my_img_sym_basic[nHard * widthBoundary], int(local_rows * widthBoundary), MPI_FLOAT,
                &img_sym_basic[nHard * widthBoundary], &counts[0], &displs[0], MPI_FLOAT, 0, MPI_COMM_WORLD);
    cout << "Rank: " << my_rank << " Gathered step 1" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbose && my_rank ==0)
        cout << "is done." << endl;
    if (my_rank == 0)
    {
        //! Check memory allocation
        img_basic.resize(img_noisy.size());
        //! To avoid boundaries problem
        // copy img_sym_basic center (without boundaries) then make boundaries symmetrical
        unsigned dc_b = nHard * widthBoundary + nHard;
        unsigned dc = 0;
        for (unsigned i = 0; i < height; i++)
            for (unsigned j = 0; j < width; j++, dc++)
                img_basic[dc] = img_sym_basic[dc_b + i * widthBoundary + j];

        makeSymmetrical(img_basic, img_sym_basic, width, height, nHard);
    }

    //! Denoising, 2nd Step
    if (verbose && my_rank ==0)
        cout << "BM3D 2nd step...";
    for (int i = 0; i < ranks; i++)
    {
        counts[i] += int(2 * nHard * widthBoundary);
    }
    for (int i = 0; i < ranks; i++)
    {
        displs[i] -= int(nHard * widthBoundary);
    }
    MPI_Scatterv(&img_sym_basic[0], &counts[0], &displs[0], MPI_FLOAT, &my_img_sym_basic[0], counts[my_rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    cout << "Rank: " << my_rank << " Scatter step 2" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    vector<float> table_2D_img((2 * nWien + 1) * widthBoundary * kWien * kWien, 0.0f);
    vector<float> table_2D_est((2 * nWien + 1) * widthBoundary * kWien * kWien, 0.0f);
    if (my_rank == 0)
    {

        vector<float> table_2D_temp_img((2 * nWien + 1) * widthBoundary * kWien * kWien, 0.0f);
        vector<float> table_2D_temp_est((2 * nWien + 1) * widthBoundary * kWien * kWien, 0.0f);
        int rank = 0;
        unsigned rank_max = (rank + 1) * (local_rows - nHard) + nHard;
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_row = row_ind[ind_i];
            //! Update of my_table_2D
            // each CPU depends on the calculation of the previous one
            if (rank < ranks - 1)
            {
                bior_2d_process(table_2D_img, img_noisy, nWien, widthBoundary, kWien, i_row, pWien, row_ind[0],
                                row_ind.back(), lpd, hpd);
                bior_2d_process(table_2D_est, img_basic, nWien, widthBoundary, kWien, i_row, pWien, row_ind[0],
                                row_ind.back(), lpd, hpd);
                if (row_ind[ind_i + 1] >= rank_max)
                {
                    rank++;
                    MPI_Send(table_2D_temp_img.data(), int(table_2D_temp_img.size()), MPI_FLOAT, rank, 0,
                             MPI_COMM_WORLD);
                    MPI_Send(table_2D_temp_est.data(), int(table_2D_temp_est.size()), MPI_FLOAT, rank, 0,
                             MPI_COMM_WORLD);
                    rank_max = (rank + 1) * (local_rows - nHard) + nHard;
                }
            }
        }
    }
    else
    {
        MPI_Recv(table_2D_img.data(), int(table_2D_img.size()), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(table_2D_est.data(), int(table_2D_est.size()), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Rank: " << my_rank << "starting step 2" << endl;
    bm3d_2nd_step(sigma, my_img_sym_noisy, my_img_sym_basic, my_img_sym_denoised, widthBoundary, heightBoundary, nWien, kWien,
                  pWien, ranks, my_rank, int(local_rows + (2 * nHard)), table_2D_img, table_2D_est);
    cout << "Rank: " << my_rank << "finished step 2" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if (verbose && my_rank ==0)
        cout << "is done." << endl;

    for (int &count : counts)
    {
        count -= int(2 * nHard * widthBoundary);
    }
    for (int &displ : displs)
    {
        displ += int(nHard * widthBoundary);
    }
    if (my_rank == 0)
    {
        img_sym_denoised.resize(widthBoundary * heightBoundary);
    }
    MPI_Gatherv(&my_img_sym_denoised[nHard * widthBoundary], int(local_rows * widthBoundary), MPI_FLOAT,
                &img_sym_denoised[nHard * widthBoundary], &counts[0], &displs[0], MPI_FLOAT, 0, MPI_COMM_WORLD);

    //! collection of img_denoised
    // copy img_sym_denoised center (without boundaries)
    if (my_rank == 0)
    {
        img_denoised.resize(img_noisy.size());
        unsigned dc_b = nWien * widthBoundary + nWien;
        unsigned dc = 0;
        for (unsigned i = 0; i < height; i++)
            for (unsigned j = 0; j < width; j++, dc++)
                img_denoised[dc] = img_sym_denoised[dc_b + i * widthBoundary + j];
        /*------------------------------------------------------------*/
        end = MPI_Wtime();
        cout << "Time: " << end - start << "s" << endl;

        //! save noisy, denoised and differences images
        cout << endl
             << "Save images...";
        char r[10];
        std::sprintf(r, "_%d.png", my_rank);
        // std::sprintf(r, "_%d.png", 0);

        if (save_image(strcat(argv[3], r), img_denoised, width, height) != EXIT_SUCCESS)
            return EXIT_FAILURE;

        cout << "done." << endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
