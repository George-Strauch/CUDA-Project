#include <iostream>
#include <omp.h>
/*
Written by George Strauch on 5/02/2020

c++ program for matrix multiply using 1d arrays.
A one dimension array is used over 2d to make better use of cache.
This implementation only uses square matrices since they are much
easier to debug, calculate and work with however all functions can work with
non-square matrices too.

for this program the matricies that are multiplied have rows 0 to (n-1), and its
transpose because this ends up making every element in the result equal to
sum[0 to (n-1)]: n^2
which makes it much easier to tell if it is working right

this uses open mp to make the program run in parallel. The inner loop
in sum_row(), made to run in parallel by openmp is the only difference from the
original cpp mm program but brings a massive time imporvement

Execution syntax:
$ ./exec {int matrix_size} {int print_option}
where the print option can be:

1: Prints the whole of each matrix for debugging
and best used with smaller matrices <= 10.
2: Shows only the first and last element of the result.
other or no option: does not print anything.

Example run:
$ g++ omp_mm.cpp -O2 -std=c++17 -fopenmp -p -o omp_mm
$ time ./omp_mm 10 1
$ time ./omp_mm 1500 2

to profile:
$ perf record -e instructions,cache-misses,cache-references,context-switches,task-clock -F 20 --call-graph dwarf ./omp_mm 1500 2
*/



// struct to make working with matrices much easier
struct Matrix
{
  long long int *values;
  int rows;
  int cols;
};



// allocates memory for the matrix
void init_matrix(Matrix &m)
{
  m.values = (long long int*)malloc(m.rows*m.cols*sizeof(long long int));
}



// builds the matrix for testing
void fill_mat(Matrix m)
{
  for (size_t i = 0; i < m.rows*m.cols; i++) {
    m.values[i] = i%m.cols;
  }
}



// gets a single element of the result and returns it
int sum_row(Matrix m1, Matrix m2, int m1_row_ind, int m2_col_ind)
{
  int s = 0;

  #pragma omp parallel for reduction(+:s)
  for (size_t i = 0; i < m1.cols; i++) {
    s += m1.values[m1_row_ind*m1.cols + i] * m2.values[i*m2.cols + m2_col_ind];
  }
  return s;
}



// multiplies m1 by m2 and returns the resulting matrix
Matrix matmul(Matrix m1, Matrix m2)
{
  // test tomake sure that sizes of m1 and m2 are compatible
  if(m1.cols != m2.rows){
    std::cout << "problem in matmul, invalid sizes" << '\n';
    exit(1);
  }

  // create the matrix to store the result that will be returned
  Matrix res;
  res.cols = m2.cols;
  res.rows = m1.rows;
  init_matrix(res);

  // every iteration of the inner loop will find one value in the result
  // outer loop gets each column vector in the result matrix
  // #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < m2.cols; i++) {
    for (size_t j = 0; j < m1.rows; j++) {
      res.values[j*res.cols + i] = sum_row(m1, m2, j, i);
    }
  }
  return res;
}



void displayMatrix(Matrix mat)
{
  for (size_t i = 0; i < mat.rows; i++) {
    for (size_t j = 0; j < mat.cols; j++) {
      std::cout << mat.values[i*mat.cols + j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}



// sets a matrix to its transpose, pass by reference
void transpose(Matrix &mat)
{
  long long int *new_mat = (long long int*)malloc(mat.rows*mat.cols*sizeof(long long int));

  for (size_t i = 0; i < mat.rows; i++) {
    for (size_t j = 0; j < mat.cols; j++) {
      new_mat[j*mat.cols+i] = mat.values[i*mat.rows+j];
    }
  }
  free(mat.values);
  mat.values = new_mat;
  int tmp = mat.rows;
  mat.rows = mat.cols;
  mat.cols = tmp;
}



// makes a copy of a matrix and returns it
Matrix copy_of_mat(Matrix m)
{
  Matrix new_mat;
  new_mat.rows = m.rows;
  new_mat.cols = m.cols;
  init_matrix(new_mat);
  for (size_t i = 0; i < m.cols*m.rows; i++) {
    new_mat.values[i] = m.values[i];
  }
  return new_mat;
}



int main(int argc, char const *argv[])
{
  // gets matrix size from user, see header
  Matrix m1;
  m1.rows = atoi(argv[1]);
  m1.cols = atoi(argv[1]);
  init_matrix(m1);
  fill_mat(m1);

  std::cout << "rows: " << m1.rows << '\n';
  std::cout << "cols: " << m1.cols << "\n\n";

  // the second matrix is just a copy of the first
  Matrix m2 = copy_of_mat(m1);
  transpose(m2);

  // performs matrix multiply
  std::cout << "start" << '\n';
  Matrix result = matmul(m1, m2);
  std::cout << "done\n" << '\n';

  // options to print the matricies
  // possible options at the top
  if (argc > 2) {
    if (atoi(argv[2]) == 1) {
      std::cout << "m1:" << '\n';
      displayMatrix(m1);
      std::cout << '\n';

      std::cout << "m2:" << '\n';
      displayMatrix(m2);
      std::cout << '\n';

      std::cout << "result: " << '\n';
      displayMatrix(result);
    }
    else if (atoi(argv[2]) == 2) {
      std::cout << "first: " << result.values[0] << '\n';
      std::cout << "last: " << result.values[result.cols*result.rows-1] << '\n';
      std::cout << '\n';
    }
  }

  // frees the memory
  std::cout << "freeing memory" << '\n';
  free(m1.values);
  free(m2.values);
  free(result.values);

  return 0;
}






















//
