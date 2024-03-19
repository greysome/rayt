#ifndef _LBVH_NODE_C
#define _LBVH_NODE_C

#include <stdbool.h>
#include "aabb.c"

typedef struct {
  aabb aabb;
  bool is_leaf;
  int idx_obj; // Index into sorted object list (by Morton code)
  int idx_split; // Index at which to split objects into two subnodes
  int idx_left; // Index of left subnode in node list
  int idx_right; // Index of right subnode in node list
} lbvh_node;

lbvh_node leaf_node(int idx_obj) {
  return (lbvh_node){.is_leaf = true, .idx_obj = idx_obj};
}

lbvh_node intermediate_node(int idx_split, int idx_left, int idx_right) {
  return (lbvh_node){.is_leaf = false, .idx_split = idx_split, .idx_left = idx_left, .idx_right = idx_right};
}

#endif