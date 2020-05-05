#include<iostream>

// edges is an array of edges where every even index is the start
// vertex and the next (odd) index is the end vertex
struct Graph {
    int* edges;
    int num_edges;
    int num_verticies;
};



Graph gen_shared_random_graph(int num_verticies, int num_edges)
{
  int max_edges = ((num_verticies-1)(num_verticies-2))/2;  // complete Graph
  num_edges = std::min(max_edges, num_edges);
  Graph *g;
  cudaMallocManaged(&g, sizeof(Matrix));
  cudaMallocManaged(&g_>edges, 2*num_edges*sizeof(int));
  g->num_edges = num_edges;
  g->num_verticies = num_verticies

  int start = 0;
  int end = 0;
  int itter = 0;
  bool again = true;

  for (size_t i = 0; i < num_edges; i++) {

    start = rand() % num_verticies;
    end = rand() % num_verticies;

    while(again) {

      start = std::min(start, end);
      end = std::max(start, end);

      if(start != end) { continue; }
      for (size_t j = 0; j < num_verticies; j++) {

        // if not (start, end) edge alread exists, and edge is not loop:
        if(
           (start == g->edges[2*j] && end == g->edges[2*j +1]) ||
           (start != end)) {
             g->edges[2*i] = start;
             g->edges[2*i + 1] = end;
             again = false;
             break;

        } else {
          start = start++ % num_verticies;
        }

      }
      // if current start is connected to all other edges:
      end++;
    }
  }
  return *g;
}




int main(int argc, char const *argv[]) {
  std::vector<int> v;

  v.push_back(1);

  std::cout << v.back() << '\n';


  return 0;
}
