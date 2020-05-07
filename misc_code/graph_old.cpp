#include<iostream>
#include <bitset>
#include<vector>

// edges is an array of edges where every even index is the start
// vertex and the next (odd) index is the end vertex
struct Graph {
    int* edges;
    int num_edges;
    int num_verticies;
};


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



// expects ba size to = g.num_edges
bool is_matching(Graph g, bool* ba)
{

  int *known = (int*)malloc(2*g.num_edges*sizeof(int));
  int index = 0;
  for (int i = 0; i < g.num_edges; i++) {
    if(ba[i])
    {
      for (size_t j = 0; j < index; j++) {
        if(g.edges[2*i] == known[j] || g.edges[2*i+1] == known[j]) {
          free(known);
          return false;
        } // end if
      }// end inner for

      known[index] = g.edges[2*i];
      known[index+1] = g.edges[2*i+1];
      index+=2;
    }// end if
  }// end for

  free(known);
  return true;
}



void graph_print(Graph g) {
  for (size_t v = 0; v < g.num_verticies; v++) {
    std::cout << "vertex " << v << " --> ";
    for (size_t i = 0; i < g.num_edges; i++) {
      if (g.edges[2*i] == v) {
        std::cout << g.edges[2*i+1] << ", ";
      }
    }
    std::cout << '\n';
  }
}



Graph gen_random_graph(int num_verticies, int num_edges)
{
  int max_edges = (num_verticies*(num_verticies-1))/2;  // complete Graph
  num_edges = std::min(max_edges, num_edges);
  std::cout << "max: " << max_edges << " assigned: " << num_edges << '\n';

  Graph g;
  g.num_edges = num_edges;
  g.num_verticies = num_verticies;
  g.edges = (int*)malloc(2*num_edges*sizeof(int));

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
        if(start == g.edges[2*j] && end == g.edges[2*j +1]) {
          end = (end+1) % num_verticies;
          again = true;
          break;
        }

        // reverse edge not already exists
        if(end == g.edges[2*j] && start == g.edges[2*j +1]) {
          end = (end+1) % num_verticies;
          again = true;
          break;
        }
      }

      if (!again) {
        g.edges[2*i] = start;
        g.edges[2*i + 1] = end;
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
    if (g.edges[2*k] > g.edges[2*k+1]) {
      tmp = g.edges[2*k];
      g.edges[2*k] = g.edges[2*k+1];
      g.edges[2*k+1] = tmp;
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
  g.edges = (int*)malloc(2*g.num_edges*sizeof(int));

  int *start = (int*)malloc(3*n*sizeof(int));
  int *end = (int*)malloc(3*n*sizeof(int));

  for (size_t i = 0; i < n; i++) {
    start[i] = i;
    end[i] = (i+1) % n;
  }

  for (size_t i = 0; i < n; i++) {
    start[i+n] = i;
    end[i+n] = i+n;
  }

  for (size_t i = 0; i < n; i++) {
    start[i+2*n] = i+n;
    end[i+2*n] = (i+1) % n;
  }

  for (size_t i = 0; i < 3*n; i++) {
    g.edges[2*i] = start[i];
    g.edges[2*i+1] = end[i];
  }
  return g;
}



int main(int argc, char const *argv[]) {
  //
  // Graph g = gen_random_graph(10, 4);
  // graph_print(g);
  //
  // std::cout << '\n';

  Graph gg = gen_known_graph(atoi(argv[1]));
  graph_print(gg);

  int mathcings = 0;
  bool *b;
  for (size_t i = 0; i < 1<<gg.num_edges; i++) {
    b = int_to_bin(i, gg.num_edges);
    if (is_matching(gg, b)) {
      mathcings++;
    }
  }
  std::cout << mathcings << '\n';
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
