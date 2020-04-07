#include <iostream>


/*
written by George Strauch on 4/03/2020

c++ program for matrix multiply using 1d arrays on the GPU
the GPU makes use of parallelism to make processes like this much faster

to run:
nvcc gpu_mm.cu -o gpu
$ time ./gpu 1000 2

*/


// res_x = res cols = m2 cols  max
// res_y = res rows = m1 rows  max
// common = m1 cols and m2 rows
__global__
void matmul(long long int* m1, long long int* m2, long long int* res, int common, int res_x, int res_y)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y+ threadIdx.y;

  if(idx >= res_x || idy >= res_y) { return; }

  int id = idy*res_x + idx;
  int m1_row = res_y * idy;

  res[id] = 0;
  for (size_t i = 0; i < common; i++) {
    res[id] += m1[m1_row+i] * m2[idx + i*res_x];
  }
}



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
  int N = atoi(argv[1]);
  std::cout << "N: " << N << '\n';

  long long int *m1 = (long long int*)malloc(N*N*sizeof(long long int));
  long long int *m2 = (long long int*)malloc(N*N*sizeof(long long int));
  long long int *rs = (long long int*)malloc(N*N*sizeof(long long int));

  long long int *d_m1;
  long long int *d_m2;
  long long int *d_rs;

  cudaMalloc(&d_m1, N*N*sizeof(long long int));
  cudaMalloc(&d_m2, N*N*sizeof(long long int));
  cudaMalloc(&d_rs, N*N*sizeof(long long int));

  for (size_t a = 0; a < N; a++) {
    for (size_t b = 0; b < N; b++) {
      m1[a*N+b] = b;
      m2[a*N+b] = b;
    }
  }

  // transpose(m2, N, N);

  // test transpose
  // std::cout << "before: " << '\n';
  // displayMatrix(m2, N, N);
  // std::cout << "\nafter: " << '\n';
  // transpose(m2, N, N);
  // displayMatrix(m2, N, N);
  // std::cout << "\n" << '\n';

  cudaMemcpy(d_m1, m1, N*N*sizeof(long long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, N*N*sizeof(long long int), cudaMemcpyHostToDevice);


  //------------------------------------------
  int thdX = 16;
  int thdY = 16;
  dim3 threads_in_block(thdX, thdY);
  dim3 block_grid((N/thdX)+1, (N/thdY)+1);
  //------------------------------------------


  std::cout << "----------------------------" << '\n';
  std::cout << "gird x: " << block_grid.x << " grid y: " << block_grid.y << '\n';
  std::cout << "total threads: " << block_grid.x * block_grid.y * thdX * thdY << ", needed: " << N*N << '\n';

  std::cout << "----------------------------" << '\n';

  std::cout << "\nstart" << '\n';
  matmul<<<block_grid, threads_in_block>>>(d_m1, d_m2, d_rs, N, N, N);
  cudaMemcpy(rs, d_rs, N*N*sizeof(long long int), cudaMemcpyDeviceToHost);
  std::cout << "done\n" << '\n';


// display array
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
