#ifndef _LBVH_C
#define _LBVH_C

#include <stdlib.h>
#include "primitive.c"
#include "aabb.c"
#include "lbvh_node.c"

int node_counter = 0;
// Given n objects, there are guaranteed to be 2n-1 nodes
lbvh_node nodes[2*MAX_OBJS-1];
#pragma acc declare create(nodes[:2*MAX_OBJS-1])

aabb get_scene_aabb(Primitive *objs, int n) {
  float min_x = INFINITY, min_y = INFINITY, min_z = INFINITY,
    max_x = -INFINITY, max_y = -INFINITY, max_z = -INFINITY;
  for (int i = 0; i < n; i++) {
    v3 pos = objs[i].pos;
    if (pos.x < min_x) min_x = pos.x;
    if (pos.x > max_x) max_x = pos.x;
    if (pos.y < min_y) min_y = pos.y;
    if (pos.y > max_y) max_y = pos.y;
    if (pos.z < min_z) min_z = pos.z;
    if (pos.z > max_z) max_z = pos.z;
  }
  return aabb_pad((aabb){(v3){min_x-1, min_y-1, min_z-1},
			 (v3){max_x+1, max_y+1, max_z+1}});
}

v3 get_relative_coords(Primitive obj, aabb scene_aabb) {
  return divi(sub(obj.pos, scene_aabb.min_coords),
	      sub(scene_aabb.max_coords, scene_aabb.min_coords));
}

// Expand a 10-bit integer into 30 bits by inserting 2 zeros after each bit
unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001) & 0xFF0000FF;
  v = (v * 0x00000101) & 0x0F00F00F;
  v = (v * 0x00000011) & 0xC30C30C3;
  v = (v * 0x00000005) & 0x49249249;
  return v;
}

// Compute a 30-bit Morton code, where pos in [0,1]^3.
unsigned int morton_code(v3 pos) {
  // We have 10 bits of precision for each coordinate. That is 1024 possible values.
  // Convert floating point to corresponding value in 0..1023
  unsigned int x = (unsigned int) clampi(pos.x*1024,0,1023);
  unsigned int y = (unsigned int) clampi(pos.y*1024,0,1023);
  unsigned int z = (unsigned int) clampi(pos.z*1024,0,1023);
  unsigned int xx = expand_bits(x);
  unsigned int yy = expand_bits(y);
  unsigned int zz = expand_bits(z);
  return xx*4 + yy*2 + zz;
}

// TODO understand what this does
// I ripped this off one of the links from README.md
int get_split(int idx_left, int idx_right) {
  // Identical Morton codes => split the range in the middle.
  unsigned int first_code = objs[idx_left].morton_code;
  unsigned int last_code = objs[idx_right].morton_code;

  if (first_code == last_code)
    return ((idx_left + idx_right)>>1) + 1;

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.

#if FOR_GPU == 0
  int common_prefix = __builtin_clz(first_code ^ last_code);
#else
  int common_prefix = __clz(first_code ^ last_code);
#endif

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.

  int split = idx_left; // initial guess
  int step = idx_right - idx_left;

  do {
      step = (step + 1) >> 1; // exponential decrease
      int new_split = split + step; // proposed new position

      if (new_split < idx_right) {
          unsigned int split_code = objs[new_split].morton_code;
#if FOR_GPU == 0
          int split_prefix = __builtin_clz(first_code ^ split_code);
#else
          int split_prefix = __clz(first_code ^ split_code);
#endif

          if (split_prefix > common_prefix)
              split = new_split; // accept proposal
      }
  }
  while (step > 1);

  return split+1;
}

int compare_morton_codes(const void *obj1_, const void *obj2_) {
  Primitive *obj1 = (Primitive *) obj1_;
  Primitive *obj2 = (Primitive *) obj2_;
  if (obj1->morton_code < obj2->morton_code)
    return -1;
  else if (obj1->morton_code > obj2->morton_code)
    return 1;
  return 0;
}

void set_node(int idx, int idx_left, int idx_right) {
  static int last_used_idx = 0;

  if (idx_left == idx_right) {
    nodes[idx] = leaf_node(idx_left);
    return;
  }

  int k = get_split(idx_left, idx_right);
  int idx_left_subnode = last_used_idx + 1,
    idx_right_subnode = last_used_idx + 2;
  nodes[idx] = intermediate_node(k, idx_left_subnode, idx_right_subnode);
  last_used_idx += 2;

  set_node(idx_left_subnode, idx_left, k-1);
  set_node(idx_right_subnode, k, idx_right);
}

void set_aabb(int node_idx) {
  lbvh_node node = nodes[node_idx];
  if (node.is_leaf)
    nodes[node_idx].aabb = get_aabb(objs[node.idx_obj]);
  else {
    set_aabb(node.idx_left);
    set_aabb(node.idx_right);
    lbvh_node node_l = nodes[node.idx_left];
    lbvh_node node_r = nodes[node.idx_right];
    nodes[node_idx].aabb = aabb_union(node_l.aabb, node_r.aabb);
  }
}

void build_lbvh() {
  // Sort objects by their morton code
  aabb scene_aabb = get_scene_aabb(objs, n_objs);
  for (int i = 0; i < n_objs; i++)
    objs[i].morton_code = morton_code(get_relative_coords(objs[i], scene_aabb));
  qsort(objs, n_objs, sizeof(Primitive), compare_morton_codes);

  // Add nodes recursively, starting with the root node at index 0
  set_node(0, 0, n_objs-1);

  // Set the nodes' aabb's recursively
  set_aabb(0);
}

void get_closest_intersection(v3 origin, v3 dir, HitRecord *hr) {
  // The values which will be written to the pointers
  float t_closest = INFINITY;
  Primitive obj_closest;
  float u_closest = 0, v_closest = 0;

  // Internal variables
  lbvh_node stack[64];
  lbvh_node *stack_ptr = stack;
  // Set the first element of the stack to be a 'NULL value'
  *stack_ptr++ = (lbvh_node){.idx_obj = -1};
  lbvh_node node = nodes[0];

  do {
    lbvh_node child_l = nodes[node.idx_left];
    lbvh_node child_r = nodes[node.idx_right];
    bool intersects_l = aabb_intersects(child_l.aabb, origin, dir);
    bool intersects_r = aabb_intersects(child_r.aabb, origin, dir);

    if (intersects_l && child_l.is_leaf) {
      Primitive obj = objs[child_l.idx_obj];
      float u, v;
      float t = get_intersection(obj, origin, dir, &u, &v);
      if (t > 0 && t < t_closest) {
	t_closest = t;
	obj_closest = obj;
	u_closest = u;
	v_closest = v;
      }
    }
    if (intersects_r && child_r.is_leaf) {
      Primitive obj = objs[child_r.idx_obj];
      float u, v;
      float t = get_intersection(obj, origin, dir, &u, &v);
      if (t > 0 && t < t_closest) {
	t_closest = t;
	obj_closest = obj;
	u_closest = u;
	v_closest = v;
      }
    }

    bool traverse_l = intersects_l && !child_l.is_leaf;
    bool traverse_r = intersects_r && !child_r.is_leaf;
    if (!traverse_l && !traverse_r)
      node = *--stack_ptr;
    else {
      node = traverse_l ? child_l : child_r;
      if (traverse_l && traverse_r)
	*stack_ptr++ = child_r;
    }
  }
  while (node.idx_obj != -1);

  hr->t = t_closest;
  hr->p = ray_at(origin, dir, t_closest);
  hr->u = u_closest;
  hr->v = v_closest;
  hr->obj = obj_closest;
}

#endif