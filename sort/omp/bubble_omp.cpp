#include <iostream>

/*
written by George Strauch on 4/21/2020

c++ program to sort an array with quicksort algorithm

Execution follows the syntax:
$ ./exec {int num of elements}

Example run:
$ g++ bubble.cpp -std=c++17 -O2 -g -fopenmp -o b
$ time ./b 10
$ time ./b 99999
*/



// used for debug
void print_array(int *array, int n)
{
  for (size_t i = 0; i < n; i++) {
    std::cout << array[i] << ' ';
  }
  std::cout << '\n';
}



// makes and returns unsorted array with random elements
int* make_unsorted_array(int n_elements)
{
  int *a = (int*)malloc(n_elements*sizeof(int));
  for (size_t j = 0; j < n_elements; j++) {
    a[j] = rand()%(2*n_elements);
  }
  return a;
}





void sort(int* array, int n)
{
  int tmp = 0;
  bool done = false;

  while (!done) {
    done = true;
    #pragma omp parallel
    for (size_t i = 1; i < n; i++) {
      if (array[i-1] > array[i]) {
        done = false;
        tmp = array[i-1];
        array[i-1] = array[i];
        array[i] = tmp;
      }
    }
  }

}



// retruuns false if any element larger than i+1 element
bool verify_in_order(int* array, int n)
{
  bool to_ret = true;
  #pragma omp parallel
  for (size_t i = 0; i < n-1; i++) {
    if (array[i+1] < array[i]) {
      std::cout << "\nindex: " << i << '\n';
      to_ret = false;
      break;
    }
  }
  return to_ret;
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

  free(a);
  return 0;
}


















//
