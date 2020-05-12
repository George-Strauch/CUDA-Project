#include <iostream>

/*
written by George Strauch on 4//2020

c++ program for matrix multiply using 1d arrays on the GPU
the GPU makes use of parallelism to make processes like this much faster

This implementation only uses square matrices as they are much
easier to debug, calculate and work with, however all functions can work with
non-square matrices too.

This program uses shared memory between the host CPU and the GPU.
using dedicated device memory can make the program run faster however its
make it much more difficult to work with certain datatypes such as the
struct to represent a matrix.

Execution syntax:
$ ./exec {int matrix_size} {int print_option}
where the print option can be:
1: Prints the whole of each matrix for debugging
and best used with smaller matrices <= 10.
2: Shows only the first and last element of the result.
other or no option: does not print anything.

Example run:
$ nvcc gpu_mm_pointers.cu -o gpu
$ time ./gpu 10 1
$ time ./gpu 1500 2
*/


typedef long long int lli;

// struct to make working with matrices much easier
struct Matrix
{
  lli *values;
  int rows;
  int cols;
};



// fills a matrix with values
__host__
void fillMat(Matrix *m)
{
  for (size_t j = 0; j < m->rows*m->cols; j++) {
    m->values[j] = j% m->cols;
  }
}



// get a Matrix object with shared memory that can be accessed by the device
__host__
Matrix* get_shared(int rows, int cols)
{
  Matrix *m;
  cudaMallocManaged(&m, sizeof(Matrix));
  cudaMallocManaged(&m->values, rows*cols*sizeof(lli));
  m->cols = cols;
  m->rows = rows;
  std::cout << "m; " << m << '\n';
  return m;
}


// calculate a single element of the matrix result of m1*m2
// res_x = res cols = m2 cols  max
// res_y = res rows = m1 rows  max
// common = m1_cols and m2_rows
__global__
void matmul(Matrix *m1, Matrix *m2, Matrix *res)
{
  // gets the id for the x and y position within the result
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;

  // there will be some invalid calls of this fuction
  // this checks to make sure this is calculating a valid element
  if(idx >= res->cols || idy >= res->rows) { return; }

  int id = idy*res->cols + idx;
  res->values[id] = 0;
  for (size_t i = 0; i < m1->cols; i++) {
    res->values[id] += m1->values[(idy*m1->cols)+i] * m2->values[(i*m2->cols)+idx];
  }
}



// host side function to display matrix
__host__
void displayMatrix(Matrix *mat)
{
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      std::cout << mat->values[i*mat->cols + j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}



// frees memory
__host__
void free_matrix(Matrix *mat)
{
  cudaFree(mat->values);
  cudaFree(mat);
}



// returns a copy of a matrix
__host__
Matrix* copyMatrix(Matrix *m)
{
  Matrix *nm = get_shared(m->rows, m->cols);
  for (size_t i = 0; i < m->cols*m->rows; i++) {
    nm->values[i] = m->values[i];
  }
  return nm;
}



// host side function to transpose
__host__
void transpose(Matrix *&mat)
{
  Matrix *new_mat = get_shared(mat->cols, mat->rows);
  for (size_t a = 0; a < mat->rows; a++) {
    for (size_t b = 0; b < mat->cols; b++) {
      new_mat->values[b*mat->cols + a] = mat->values[a*mat->cols + b];
    }
  }

  free_matrix(mat);
  mat = new_mat;
}



int main(int argc, char const *argv[])
{
  // gets the matrix size from user, see header
  int N = atoi(argv[1]);
  std::cout << "N: " << N << '\n';

  Matrix *t1 = get_shared(N, N);
  std::cout << "t1: " << &t1 << '\n';
  fillMat(t1);
  Matrix *t2 = copyMatrix(t1);
  transpose(t2);


  Matrix *res = get_shared(t1->rows, t2->cols);


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
  matmul<<<block_grid, threads_in_block>>>(t1, t2, res);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  std::cout << "done\n" << '\n';

  // display array
  // display options listed in header
  if (argc > 2) {
    if (atoi(argv[2]) == 1) {
      std::cout << "matrix 1: " << '\n';
      displayMatrix(t1);

      std::cout << "matrix 2: " << '\n';
      displayMatrix(t2);

      std::cout << "result: " << '\n';
      displayMatrix(res);
    }
    else if (atoi(argv[2]) == 2) {
      std::cout << "first: " << res->values[0] << '\n';
      std::cout << "last: " << res->values[N*N-1] << '\n';
      std::cout << '\n';
    }
  }

  free_matrix(t1);
  free_matrix(t2);
  free_matrix(res);

  return 0;

}














//
