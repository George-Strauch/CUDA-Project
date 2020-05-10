#include <iostream>
#include <cuda.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

/*
written by George Strauch on 4/21/2020

c++ program to sort an array with quicksort algorithm

Execution follows the syntax:
$ ./exec {int num of elements}

Example run:
$ nvcc gpu_bubble.cu -arch='sm_35' -rdc=true -lineinfo  -lcudadevrt -o gpu_q
$ time ./gpu_qs 10
$ time ./gpu_qs 999
*/



__host__  // used for debug
void print_array (int *array, int n, int tag_index)
{
  for (size_t i = 0; i < n; i++) {
    if (i == tag_index+1) {
      std::cout << " > ";
    }
    std::cout << array[i] << ' ';
  }
  std::cout << '\n';
}



__host__
int* allocate_shared_array(int n_elements)
{
  int *a;
  cudaMallocManaged(&a, n_elements*sizeof(int));
  return a;
}



__host__  // makes and returns unsorted array with random elements
int* make_unsorted_array(int n_elements)
{
  int *a = allocate_shared_array(n_elements);
  for (size_t j = 0; j < n_elements; j++) {
    a[j] =  rand()%(2*n_elements);
  }
  return a;
}



__host__
bool go_again(int* array, int n)
{
  for (size_t i = 0; i < n-1; i++) {
    if(array[i] > array[i+1])
    {
      return true;
    }
  }
  return false;
}



__global__
void sort(int* array, int n, int offset, int k)
{
  int id = 2*(blockIdx.x*blockDim.x + threadIdx.x) + offset + k;
  if (id >= n) {
    return;
  }


  int tmp;
  if (array[id] > array[id+1]) {
    tmp = array[id+1];
    array[id+1] = array[id];
    array[id] = tmp;
  }
  __syncthreads();

}


__host__
void fill_array(int* a, int n) {
  for (size_t i = 0; i < n; i++) {
    a[i] = 0;
  }
}



__host__  // returns element index if any element larger than i+1 element, else -1
int verify_in_order(int* array, int n)
{
  for (size_t i = 0; i < n-1; i++) {
    if (array[i+1] < array[i]) {
      return i;
    }
  }
  return -1;
}


__host__
void entry_point(int* array, int n)
{
  int t = 512;
  int b = 512;
  int count = 0;
  int total = t*b;
  dim3 threads(t);
  dim3 blocks(b);
  // fill_array(array, n);
  while (go_again(array, n)) {
    cudaDeviceSynchronize();
    for (size_t i = 0; i < (n/(2*total)) +1; i++) {
      sort<<<blocks, threads>>>(array, n, 0, i*total);
      cudaDeviceSynchronize();
      sort<<<blocks, threads>>>(array, n, 1, i*total);
    }
    cudaDeviceSynchronize();
    count++;
    if (count > 1.5*n) {
      break;
    }
  }
}



int main(int argc, char const *argv[])
{

  int N = atoi(argv[1]);
  std::cout << "N = " << N << '\n';

  int* a = make_unsorted_array(N);

  cudaProfilerStart();

  // while (go_again(a, N)) {
  //   sort<<<1, N/2>>>(a, N, 0);
  //   sort<<<1, N/2>>>(a, N, 1);
  //   cudaDeviceSynchronize();
  // }
  entry_point(a, N);
  cudaProfilerStop();

  int order = verify_in_order(a, N);

  if (order == -1) {
    std::cout << "array is in order" << '\n';
  }
  else {
    std::cout << "not in order"  << '\n';
    print_array(a, N, order);
  }

  cudaFree(a);
  return 0;
}


















//
