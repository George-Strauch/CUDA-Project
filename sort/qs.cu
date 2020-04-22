#include <iostream>

/*
written by George Strauch on 4/21/2020

c++ program to sort an array with quicksort algorithm

Execution follows the syntax:
$ ./exec {int num of elements}

Example run:
$ nvcc qs_gpu -o gpu_qs
$ time ./gpu_qs 10
$ time ./gpu_qs 99999 2
*/



__host__  // used for debug
void print_array (int *array, int n)
{
  for (size_t i = 0; i < n; i++) {
    std::cout << array[i] << ' ';
  }
  std::cout << '\n';
}



__host__  // makes and returns unsorted array with random elements
int* make_unsorted_array(int n_elements)
{
  int *a;
  cudaMallocManaged(&a, n_elements*sizeof(int));
  for (size_t j = 0; j < n_elements; j++) {
    a[j] =  rand()%(2*n_elements);
  }
  return a;
}



__host__  // helper function to overwrite section of array
void overwrite(int* &new_array, int* original, int n)
{
  for (size_t i = 0; i < n; i++) {
    original[i] = new_array[i];
  }
  cudaFree(new_array);
}



__host__
void sort(int* array, int n)
{
  // dont do anything if array size is 0 or 1
  if (n < 2) { return; }

  // new array that will overwrite current array block
  int *new_array;
  cudaMallocManaged(&new_array, n*sizeof(int));

  int piv = array[n-1];
  int lower_or_equal = 0;   // num of elements lower or equal to piv
  int higher = 0;           // num of elements higher to piv

  // if element lower or equal to piv, append to bottom of new array
  // else, apend to top
  // then overwite array with new_array
  for (size_t i = 0; i < n; i++) {
    if(array[i] <= piv){
      new_array[lower_or_equal] = array[i];
      lower_or_equal++;
    }
    else {
      new_array[n-higher-1] = array[i];
      higher++;
    }
  }
  overwrite(new_array, array, n);

  // if no elements are higher than piv, piv remains at top, so sort bottom n-1
  if (higher == 0) {
    sort(array, lower_or_equal-1);
  }
  else {
    sort(array, lower_or_equal);
    sort(&array[lower_or_equal], higher);
  }
}



__host__  // retruuns false if any element larger than i+1 element
bool verify_in_order(int* array, int n)
{
  for (size_t i = 0; i < n-1; i++) {
    if (array[i+1] < array[i]) {
      std::cout << "\nindex: " << i << '\n';
      return false;
    }
  }
  return true;
}



int main(int argc, char const *argv[])
{
  int N = atoi(argv[1]);
  std::cout << "N = " << N << '\n';

  int* a = make_unsorted_array(N);
  sort(a,N);

  if (verify_in_order(a, N)) {
    std::cout << "array is in order" << '\n';
  }
  else {
    std::cout << "not in order"  << '\n';
  }

  cudaFree(a);
  return 0;
}


















//
