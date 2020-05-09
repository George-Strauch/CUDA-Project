#include<iostream>
#include <bitset>
#include<vector>

// edges is an array of edges where every even index is the start
// vertex and the next (odd) index is the end vertex
struct Graph {
    int* edges_start;
    int* edges_end;
    int num_edges;
    int num_verticies;
};






__host__
int* allocate_device_array(int n_elements)
{
  int *a;
  cudaMalloc(&a, n_elements*sizeof(int));
  return a;
}


__host__
int* allocate_host_array(int n_elements)
{
  return (int*)malloc(n_elements*sizeof(int));
}



__host__
int* get_elements(int* &d_array, int n)
{
  int *a = allocate_host_array(n);
  cudaMemcpy(a, d_array, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_array);
  return a;
}




__device__
void concat_bool_array(bool* array1, int len1, bool* array2, int len2, bool* result){

  for (size_t i = 0; i < len1; i++) {
    result[i] = array1[i];
  }

  for (size_t i = 0; i < len2; i++) {
    result[i+len1] = array2[i];
  }
}




void print_bin(bool* b, int len) {
  for (size_t i = 0; i < len; i++) {
    std::cout << b[i];
  }
  std::cout << '\n';
}



bool* int_to_bin(int num, int len)
{

  bool* ba = (bool*)malloc(len*sizeof(bool));

  for (int i = 0; i < len; ++i) {
    // ba[len-i-1] = num%2;
    ba[i] = num%2;
    num /= 2;
  }
  return ba;
}


__device__
void int_to_bin_fill(int num, int len, bool* array)
{
  for (int i = 0; i < len; ++i) {
    // ba[len-i-1] = num%2;
    array[i] = num%2;
    num /= 2;
  }
}



// // expects ba size to = g.num_edges
// bool is_matching(Graph g, bool* ba)
// {
//   int *known = (int*)malloc(2*g.num_edges*sizeof(int));
//   int index = 0;
//   for (int i = 0; i < g.num_edges; i++) {
//     if(ba[i])
//     {
//       for (size_t j = 0; j < index; j++) {
//         if(g.edges_start[i] == known[j] || g.edges_end[i] == known[j]) {
//           free(known);
//           return false;
//         } // end if
//       }// end inner for
//
//       known[index] = g.edges_start[i];
//       known[index+1] = g.edges_end[i];
//       index+=2;
//     }// end if
//   }// end for
//
//   free(known);
//   return true;
// }



// expects ba size to = g.num_edges
__device__
bool is_matching(Graph g, bool* ba)
{
  int *known;
  cudaMalloc(&known, 2*g.num_edges*sizeof(int));
  int index = 0;
  for (int i = 0; i < g.num_edges; i++) {
    if(ba[i])
    {

      for (size_t j = 0; j < index; j++) {

        if(g.edges_start[i] == known[j] || g.edges_end[i] == known[j]) {
          free(known);
          return false;
        } // end if

      }// end inner for

      known[index] = g.edges_start[i];
      known[index+1] = g.edges_end[i];
      index+=2;
    }// end if
  }// end for

  cudaFree(known);
  return true;
}





void graph_print(Graph g) {
  for (size_t v = 0; v < g.num_verticies; v++) {
    std::cout << "vertex " << v << " --> ";
    for (size_t i = 0; i < g.num_edges; i++) {
      if (g.edges_start[i] == v) {
        std::cout << g.edges_end[i] << ", ";
      }
    }
    std::cout << '\n';
  }
}



__global__
void find_n_matchings(Graph g, int* num_matchings, int instances)
{

  // extern __shared__ bool num_of_matchings[];
  int id = threadIdx.x;
  int iter = 1;
  int len = g.num_edges-instances;
  bool* first_elements;
  bool* next_elements;
  bool* to_test;
  cudaMalloc(&to_test, (g.num_edges)*sizeof(bool));
  cudaMalloc(&first_elements, (instances)*sizeof(bool));
  cudaMalloc(&next_elements, len*sizeof(bool));

  int_to_bin_fill(id, instances, first_elements);
  int_to_bin_fill(0, len, next_elements);
  concat_bool_array(first_elements, instances, next_elements, len, to_test);
  __syncthreads();

  if(!is_matching(g, to_test)) {
    num_matchings[id] = 0;
    return;
  }

  for (size_t i = 1; i < (1<<len); i++) {
    int_to_bin_fill(i, len, next_elements);
    concat_bool_array(first_elements, instances, next_elements, len, to_test);
  if(is_matching(g, to_test)) {
      iter++;
      return;
    }
  }

  num_matchings[id] = iter;

  cudaFree(to_test);
  cudaFree(first_elements);
  cudaFree(next_elements);

  __syncthreads();
  return;
}



int count(int* &d_array, int len)
{
  int total = 0;
  int * elements = get_elements(d_array, len);
  for (size_t i = 0; i < len; i++) {
    total += elements[i];
  }
  cudaFree(d_array);
  free(elements);
  return total;
}



int count_matchings(Graph g)
{
  int instances = 5;
  int* d_array = allocate_device_array(g.num_edges);
  find_n_matchings<<<1,(1<<instances)>>>(g, d_array, instances);
  cudaDeviceSynchronize();
  return count(d_array, g.num_edges);
}



Graph gen_random_graph(int num_verticies, int num_edges)
{
  int max_edges = (num_verticies*(num_verticies-1))/2;  // complete Graph
  num_edges = std::min(max_edges, num_edges);
  std::cout << "max: " << max_edges << " assigned: " << num_edges << '\n';

  Graph g;
  g.num_edges = num_edges;
  g.num_verticies = num_verticies;
  g.edges_start = (int*)malloc(num_edges*sizeof(int));
  g.edges_end = (int*)malloc(num_edges*sizeof(int));

  int start, end, tmp, iter;
  bool again;

  for (size_t i = 0; i < num_edges; i++) {
    iter = 0;
    start = rand() % num_verticies;
    end = rand() % num_verticies;
    again = true;

    while(again) {
      again = false;

      for (size_t j = 0; j < num_edges; j++) {
        // validates not loop edge
        if (start == end) {
          end = (end+1) % num_verticies;
          again = true;
          break;
        }

        // edge not already exists
        if(start == g.edges_start[j] && end == g.edges_end[j]) {
          end = (end+1) % num_verticies;
          again = true;
          break;
        }

        // reverse edge not already exists
        if(end == g.edges_start[j] && start == g.edges_end[j]) {
          end = (end+1) % num_verticies;
          again = true;
          break;
        }
      }

      if (!again) {
        g.edges_start[i] = start;
        g.edges_end[i] = end;
      }
      else {
        if (iter % num_verticies == 0 && iter != 0) {
          start = (start+1) % num_verticies;
        }
        iter++;
      } // end assign edge
    } // end while
  } // end edge creation for loop

  // make the graph nicer
  for (size_t k = 0; k < num_edges; k++) {
    if (g.edges_start[k] > g.edges_end[k]) {
      tmp = g.edges_start[k];
      g.edges_start[k] = g.edges_end[k];
      g.edges_end[k] = tmp;
    }
  }

  return g;
}



// generates a predictible graph
Graph gen_known_graph(int n)
{

  Graph g;
  g.num_edges = 3*n;
  g.num_verticies = 2*n;

  g.edges_start = (int*)malloc(3*n*sizeof(int));
  g.edges_end = (int*)malloc(3*n*sizeof(int));

  // main loop
  for (size_t i = 0; i < n; i++) {
    g.edges_start[i] = i;
    g.edges_end[i] = (i+1) % n;
  }

  // main to outter
  for (size_t i = 0; i < n; i++) {
    g.edges_start[i+n] = i;
    g.edges_end[i+n] = i+n;
  }

  // outter to main
  for (size_t i = 0; i < n; i++) {
    g.edges_start[i+2*n] = i+n;
    g.edges_end[i+2*n] = (i+1) % n;
  }

  return g;
}



int main(int argc, char const *argv[]) {

  Graph gg = gen_known_graph(atoi(argv[1]));
  graph_print(gg);

  // int *test = allocate_host_array(10);
  //
  // for (size_t i = 0; i < 10; i++) {
  //   test[i] = 10;
  // }
  //
  // std::cout << count(test, ) << '\n';


  int a = count_matchings(gg);
  std::cout << a << '\n';



  return 0;
}

































//
//
// // save of gen graph with debug prints
// Graph gen_random_graph(int num_verticies, int num_edges)
// {
//   int max_edges = (num_verticies*(num_verticies-1))/2;  // complete Graph
//   num_edges = std::min(max_edges, num_edges);
//   std::cout << "max: " << max_edges << " assigned: " << num_edges << '\n';
//
//   Graph g;
//   g.num_edges = num_edges;
//   g.num_verticies = num_verticies;
//   g.edges = (int*)malloc(2*num_edges*sizeof(int));
//
//   int start, end, tmp, iter;
//   bool again;
//
//   for (size_t i = 0; i < num_edges; i++) {
//     iter = 0;
//     start = rand() % num_verticies;
//     end = rand() % num_verticies;
//     again = true;
//
//     while(again) {
//       again = false;
//
//       std::cout << "***********************" << '\n';
//       for (size_t j = 0; j < num_edges; j++) {
//
//         // validates not loop edge
//         if (start == end) {
//           std::cout << "******** s = e: " << start << " = " << end << '\n';
//           end = (end+1) % num_verticies;
//           std::cout << "******** s != e: " << start << " = " << end << '\n' << '\n';
//           again = true;
//           break;
//         }
//
//         // edge not already exists
//         if(start == g.edges[2*j] && end == g.edges[2*j +1]) {
//           std::cout << "edge exists" << '\n';
//           std::cout << start << " == " << g.edges[2*j] << " && " << end << " == " << g.edges[2*j +1] << '\n';
//           end = (end+1) % num_verticies;
//           again = true;
//           break;
//         }
//
//         // reverse edge not already exists
//         if(end == g.edges[2*j] && start == g.edges[2*j +1]) {
//           std::cout << "edge exists" << '\n';
//           std::cout << end << " == " << g.edges[2*j] << " && " << start << " == " << g.edges[2*j +1] << '\n';
//           end = (end+1) % num_verticies;
//           again = true;
//           break;
//         }
//         std::cout << start << " != " << g.edges[2*j] << " && " << end << " != " << g.edges[2*j +1] << '\n';
//       }
//       std::cout << "***********************" << '\n';
//
//       if (!again) {
//         g.edges[2*i] = start;
//         g.edges[2*i + 1] = end;
//         std::cout << "----------------------------------- done: " << start << " --> " << end << "\n\n" << '\n';
//       }
//       else {
//         if (iter % num_verticies == 0 && iter != 0) {
//           std::cout << "iterating start " << start << " = " << end << '\n';
//           start = (start+1) % num_verticies;
//         }
//         iter++;
//       } // end assign edge
//
//     } // end while
//
//   } // end edge creation for loop
//
//   // make the graph nicer
//   for (size_t k = 0; k < num_edges; k++) {
//     if (g.edges[2*k] > g.edges[2*k+1]) {
//       tmp = g.edges[2*k];
//       g.edges[2*k] = g.edges[2*k+1];
//       g.edges[2*k+1] = tmp;
//     }
//   }
//
//   return g;
// }
