#include<iostream>
#include<bitset>
#include<vector>

// edges is an array of edges where every even index is the start
// vertex and the next (odd) index is the end vertex
struct Leaf {
    Leaf* left;
    Leaf* right;
    int value;
};


Leaf* construct_tree(int n_deep)
{
  Leaf* this_root;
  if (n_deep == 0) {
    return this_root;
  }
  else {
    this_root->left = construct_tree(n_deep-1);
    this_root->right = construct_tree(n_deep-1);
    return this_root;
  }
}



int dft_count(Leaf root)
{
  std::cout << "here" << '\n';
  if (root.left == NULL || root.right == NULL) {
    return 1;
  }
  else {
    return dft_count(*root.left) + dft_count(*root.right);
  }
}


int main(int argc, char const *argv[]) {
  int deepth = 10;
  Leaf* r = construct_tree(deepth);
  Leaf root = *r;
  int c = dft_count(root);

  std::cout << c << '\n';

  return 0;
}
