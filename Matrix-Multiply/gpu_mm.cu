#include <iostream>

/*
written by George Strauch on 4/03/2020

c++ program for matrix multiply using 1d arrays on the GPU
the GPU makes use of parallelism to make processes like this much faster

This implementation only uses square matrices as they are much
easier to debug, calculate and work with, however all functions can work with
non-square matrices too.

this program does not ((yet)) use a struct to make working witht the
matricies easier as this runs into problems with CUDA however this will change

Execution follows the syntax:
$ ./exec {int matrix_size} {int print_option}
where the print option can be:
1: Prints the whole of each matrix for debugging
and best used with smaller matrices <= 10.
2: Shows only the first and last element of the result.
other or no option: does not print anything.

Example run:
$ nvcc gpu_mm.cu -o gpu
$ time ./gpu 10 1
$ time ./gpu 1500 2
*/



//calculate a single element of the matrix result of m1*m2
// res_x = res cols = m2 cols  max
// res_y = res rows = m1 rows  max
// common = m1_cols and m2_rows
__global__
void matmul(long long int* m1, long long int* m2, long long int* res, int common, int res_x, int res_y)
{
  // gets the id for the x and y position within the result
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y+ threadIdx.y;

  // there will be some invalid calls of this fuction
  // this checks to make sure this is calculating a valid element
  if(idx >= res_x || idy >= res_y) { return; }

  int id = idy*res_x + idx;
  res[id] = 0;
  for (size_t i = 0; i < common; i++) {
    res[id] += m1[res_y*idy+i] * m2[i*res_x+idx];
  }
}



// host side function to display matrix
__host__
void displayMatrix(long long int *mat, long long int rows, long long int cols)
{
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      std::cout << mat[i*cols + j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';

}



// host side function to transpose
__host__
void transpose(long long int *&mat, long long int rows, long long int cols)
{
  long long int *new_mat = (long long int*)malloc(rows*cols*sizeof(long long int));
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      new_mat[j*cols+i] = mat[i*rows+j];
    }
  }
  free(mat);
  mat = new_mat;
}



int main(int argc, char const *argv[])
{
  // gets the matrix size from user, see header
  int N = atoi(argv[1]);
  std::cout << "N: " << N << '\n';

  // host side matrices
  long long int *m1 = (long long int*)malloc(N*N*sizeof(long long int));
  long long int *m2 = (long long int*)malloc(N*N*sizeof(long long int));
  long long int *rs = (long long int*)malloc(N*N*sizeof(long long int));

  // device side matricies
  long long int *d_m1;
  long long int *d_m2;
  long long int *d_rs;

  // allocate memory for the matrices on the device
  cudaMalloc(&d_m1, N*N*sizeof(long long int));
  cudaMalloc(&d_m2, N*N*sizeof(long long int));
  cudaMalloc(&d_rs, N*N*sizeof(long long int));

  // initalizes the host matricies
  for (size_t a = 0; a < N; a++) {
    for (size_t b = 0; b < N; b++) {
      m1[a*N+b] = b;
      m2[a*N+b] = b;
    }
  }

  // transpose(m2, N, N);

  // copies host matrices to devices memory
  cudaMemcpy(d_m1, m1, N*N*sizeof(long long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, N*N*sizeof(long long int), cudaMemcpyHostToDevice);


  // options for building the block grid. Subject to Change.
  //------------------------------------------
  int thdX = 16;
  int thdY = 16;
  dim3 threads_in_block(thdX, thdY);
  dim3 block_grid((N/thdX)+1, (N/thdY)+1);
  //------------------------------------------

  // debug info about the block grid
  std::cout << "----------------------------" << '\n';
  std::cout << "gird x: " << block_grid.x << " grid y: " << block_grid.y << '\n';
  std::cout << "total threads: " << block_grid.x * block_grid.y * thdX * thdY << ", needed: " << N*N << '\n';
  std::cout << "----------------------------" << '\n';

  // performs matrix multiply on the GPU, each thread will handle one element
  // then copys the result to host memory
  std::cout << "\nstart" << '\n';
  matmul<<<block_grid, threads_in_block>>>(d_m1, d_m2, d_rs, N, N, N);
  cudaMemcpy(rs, d_rs, N*N*sizeof(long long int), cudaMemcpyDeviceToHost);
  std::cout << "done\n" << '\n';


// display array
// display options listed in header
  if (argc > 2) {
    if (atoi(argv[2]) == 1) {
      std::cout << "matrix 1: " << '\n';
      displayMatrix(m1, N, N);

      std::cout << "matrix 2: " << '\n';
      displayMatrix(m2, N, N);

      std::cout << "result: " << '\n';
      displayMatrix(rs, N, N);
    }
    else if (atoi(argv[2]) == 2) {
      std::cout << "first: " << rs[0] << '\n';
      std::cout << "last: " << rs[N*N-1] << '\n';
      std::cout << '\n';

    }
  }

  // frees device and host memeory
  std::cout << "freeing memory" << '\n';
  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_rs);

  free(m1);
  free(m2);
  free(rs);

  return 0;
}














//
