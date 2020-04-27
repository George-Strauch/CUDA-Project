#include <iostream>

/*
written by George Strauch on 4/21/2020

c++ program to sort an array with quicksort algorithm

Execution follows the syntax:
$ ./exec {int num of elements}

Example run:
$ nvcc qs.cu -arch='sm_35' -rdc=true -lcudadevrt -o gpu_q
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



__device__  // helper function to overwrite section of array
void overwrite(int* &new_array, int* original, int n)
{
  for (size_t i = 0; i < n; i++) {
    original[i] = new_array[i];
  }
  cudaFree(new_array);
}



// __global__
// void sort(int* array, int n)
// {
//   // dont do anything if array size is 0 or 1
//   if (n < 2) { return; }
//   int *tmparray;
//   cudaMalloc(&tmparray, n*sizeof(int));
//
//   int piv = array[n-1];
//   int lower_or_equal = 0;   // num of elements lower or equal to piv
//   int higher = 0;           // num of elements higher to piv
//
//   // if element lower or equal to piv, append to bottom of new array
//   // else, apend to top
//   // then overwite array with new_array
//   for (size_t i = 0; i < n; i++) {
//     if(array[i] <= piv){
//       tmparray[lower_or_equal] = array[i];
//       lower_or_equal++;
//     }
//     else {
//       tmparray[n-higher-1] = array[i];
//       higher++;
//     }
//   }
//   overwrite(tmparray, array, n);
//
//   // if no elements are higher than piv, piv remains at top, so sort bottom n-1
//   if (higher == 0) {
//     sort<<<1,1>>>(array, lower_or_equal-1);
//   }
//   else {
//     sort<<<1,1>>>(array, lower_or_equal);
//     sort<<<1,1>>>(&array[lower_or_equal], higher);
//   }
//   cudaDeviceSynchronize();
// }












__global__
void sort(int* array, int l, int h)
{
  // dont do anything if array size is 0 or 1
  if (l+h < 2) { return; }

  int n;

  if (threadIdx.x == 0){
    array = array;
    n = l;
  }
  else {
    array = &array[l];
    n = h-l;
  }

  int *tmparray;
  cudaMalloc(&tmparray, n*sizeof(int));

  int piv = array[n-1];
  int lower_or_equal = 0;   // num of elements lower or equal to piv
  int higher = 0;           // num of elements higher to piv

  // if element lower or equal to piv, append to bottom of new array
  // else, apend to top
  // then overwite array with new_array
  for (size_t i = 0; i < n; i++) {
    if(array[i] <= piv){
      tmparray[lower_or_equal] = array[i];
      lower_or_equal++;
    }
    else {
      tmparray[n-higher-1] = array[i];
      higher++;
    }
  }
  overwrite(tmparray, array, n);

  // if no elements are higher than piv, piv remains at top, so sort bottom n-1
  if (higher == 0) {
    sort<<<1,1>>>(array, lower_or_equal, higher);
  }
  else {
    sort<<<1,2>>>(array, lower_or_equal, higher);
  }

  cudaDeviceSynchronize();
}









__host__  // retruuns false if any element larger than i+1 element
int verify_in_order(int* array, int n)
{
  for (size_t i = 0; i < n-1; i++) {
    if (array[i+1] < array[i]) {
      std::cout << "\nindex: " << i << '\n';
      return i;
    }
  }
  return -1;
}



int main(int argc, char const *argv[])
{

  int N = atoi(argv[1]);
  std::cout << "N = " << N << '\n';

  int* a = make_unsorted_array(N);

  // sort(array, lower_or_equal, higher, n-1);
  sort<<<1,1>>>(a, N, 0);
  cudaDeviceSynchronize();

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
