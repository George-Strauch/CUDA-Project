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

  free(known);
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

  int mathcings = 0;
  bool *b;
  for (size_t i = 0; i < 1<<gg.num_edges; i++) {
    b = int_to_bin(i, gg.num_edges);
    if (is_matching(gg, b)) {
      mathcings++;
    }
  }
  std::cout << mathcings << '\n';






std::cout << "\n" << '\n';

Graph h = gen_random_graph(10, 30);
graph_print(h);




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
