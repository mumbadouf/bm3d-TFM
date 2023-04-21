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
  //! Variables initialization
  const bool verbose = pick_option(&argc, argv, "verbose", nullptr) != nullptr;

  const int patch_size = 8;

  //! Check if there is the right call for the algorithm
  if (argc < 4)
  {
    cerr << "usage: " << argv[0] << " input sigma output [basic]\n\
             [-verbose]"
         << endl;
    return EXIT_FAILURE;
  }

  //! Declarations
  vector<float> img_noisy, img_basic, img_denoised;
  unsigned width, height;

  //! Load image
  if (load_image(argv[1], img_noisy, &width, &height) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  float fSigma = strtof(argv[2], nullptr);
  double start = omp_get_wtime();
  //! Denoising
  if (run_bm3d(fSigma, img_noisy, img_basic, img_denoised, width, height,
               patch_size,
               verbose) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  double end = omp_get_wtime();
  cout << "Time: " << end - start << "s" << endl;

  //! save noisy, denoised and differences images
  cout << endl
       << "Save images...";

  if (save_image(argv[3], img_denoised, width, height) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cout << "done." << endl;

  return EXIT_SUCCESS;
}
