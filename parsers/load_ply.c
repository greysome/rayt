#include "load_ply.h"

typedef enum {
  BINARY_LITTLE_ENDIAN
} PlyFormat;

typedef struct {
  PlyFormat fmt;
  int nvertices;
  int nvertexfields;
  int nfaces;
  // Set to true if vertices are listed before faces
  // It's called funny because of how I update its value
  bool funny;
  int xpos, ypos, zpos, nxpos, nypos, nzpos, upos, vpos;
} PlyHeader;

#define EOL  if (**contents != '\n') ERROR("load_ply: expected end-of-line") else (*contents)++;
static void parse_header(char **contents, PlyHeader *header) {
  // This will be populated while parsing
  header->funny = false;
  header->nvertexfields = 0;
  header->xpos = header->ypos = header->zpos = header->nxpos = header->nypos = header->nzpos = header->upos = header->vpos = -1;

  int element_type;
  int field_pos;

  // First line
  if (!consume(contents, "ply"))
    ERROR("load_ply: not a .ply file");
  DEBUG("load_ply: valid magic header");
  next_line(contents);

  // The remaining lines
header_global:
  if (consume(contents, "comment")) {
    DEBUG("load_ply: comment");
    next_line(contents);
    goto header_global;
  }

  else if (consume(contents, "format")) {
    DEBUG("load_ply: format");
    if (consume(contents, "binary_little_endian")) {
      header->fmt = BINARY_LITTLE_ENDIAN;
    }
    else ERROR("load_ply: format: invalid format");

    // Ignore version number
    if (**contents != '\n') swallow(contents);
    EOL;

    goto header_global;
  }

  else if (consume(contents, "element")) {
E:
    DEBUG("load_ply: element");

    if (consume(contents, "vertex")) {
      header->funny = !header->funny;
      element_type = 0;
      if (!consume_int(contents, &header->nvertices))
        ERROR("load_ply: element: vertex: expected an integer");
      DEBUG("load_ply: element: vertex: %d", header->nvertices);
    }
    else if (consume(contents, "face")) {
      if (!header->funny) header->funny = true;
      element_type = 1;
      if (!consume_int(contents, &header->nfaces))
        ERROR("load_ply: element: face: expected an integer");
      DEBUG("load_ply: element: face: %d", header->nfaces);
    }
    else ERROR("load_ply: element: unknown element");

    EOL;
    field_pos = 0;
    goto header_element;
  }

  else if (consume(contents, "end_header")) {
    EOL;
    return;
  }

  else ERROR("load_ply: unknown keyword");

header_element:
  if (consume(contents, "comment")) {
    DEBUG("load_ply: comment");
    next_line(contents);
    goto header_element;
  }

  else if (consume(contents, "property")) {
    DEBUG("load_ply: property");
    if (consume(contents, "float")) {
      if (element_type == 0) {
        if (consume(contents, "x")) header->xpos = field_pos;
        else if (consume(contents, "y")) header->ypos = field_pos;
        else if (consume(contents, "z")) header->zpos = field_pos;
        else if (consume(contents, "nx")) header->nxpos = field_pos;
        else if (consume(contents, "ny")) header->nypos = field_pos;
        else if (consume(contents, "nz")) header->nzpos = field_pos;
        else if (consume(contents, "u")) header->upos = field_pos;
        else if (consume(contents, "v")) header->vpos = field_pos;
        else DEBUG("element: property: float: ignoring property");
        header->nvertexfields++;
      }
      else if (element_type == 1) {
        ERROR("element: property: float: face property should be a list");
      }
    }

    else if (consume(contents, "list")) {
      DEBUG("load_ply: list");
      if (consume(contents, "uint8")) {
      	if (consume(contents, "int")) {
          // Assume we are referring to vertex_indices
          swallow(contents);
      	}
      	else ERROR("load_ply: property: list: invalid index type");
      }
      else ERROR("load_ply: property: list: invalid index count type");
    }

    else ERROR("load_ply: property: invalid type");

    EOL;
    field_pos++;
    goto header_element;
  }

  else if (consume(contents, "element")) {
    goto E;
  }

  else if (consume(contents, "end_header")) {
    DEBUG("load_ply: end_header");
    EOL;
    return;
  }

  else {
    ERROR("load_ply: unknown keyword");
  }
}
#undef EOL

static void parse_vertices(char **contents, PlyHeader *header, Model *model) {
  for (int i = 0; i < header->nvertices; i++) {
    Vertex vertex;
    float x, y, z, nx, ny, nz, u, v;
    for (int field_pos = 0; field_pos < header->nvertexfields; field_pos++) {
      float f = *((float*)*contents);
      *contents += sizeof(float);
      if (field_pos == header->xpos) vertex.x = f;
      else if (field_pos == header->ypos) vertex.y = f;
      else if (field_pos == header->zpos) vertex.z = f;
      else if (field_pos == header->nxpos) vertex.nx = f;
      else if (field_pos == header->nypos) vertex.ny = f;
      else if (field_pos == header->nzpos) vertex.nz = f;
      else if (field_pos == header->upos) vertex.u = f;
      else if (field_pos == header->vpos) vertex.v = f;
    }
    arrpush(model->arr_vertices, vertex);
  }
}

static void parse_faces(char **contents, PlyHeader *header, Model *model) {
  for (int i = 0; i < header->nfaces; i++) {
    int nsides = *((*contents)++);
    int *idxs = malloc(nsides * sizeof(int));
    for (int j = 0; j < nsides; j++) {
      idxs[j] = *((int*)*contents);
      *contents += sizeof(int);
    }

    _Face face;
    face.nsides = nsides;
    face.idxs = idxs;
    arrpush(model->arr_faces, face);
  }
}

void load_ply(char *filename, Model *model) {
  char *contents = read_whole_file(filename);
  char *contents_start = contents;

  PlyHeader header;
  parse_header(&contents, &header);

  if (header.xpos == -1 || header.ypos == -1 || header.zpos == -1 || header.nxpos == -1 || header.nypos == -1 || header.nzpos == -1 || header.upos == -1 || header.vpos == -1)
    ERROR("load_ply: not all vertex fields (x,y,z,nx,ny,nz,u,v) are defined");
  printf("[rayt] load_ply: header: fmt = %d\n", header.fmt);
  printf("[rayt] load_ply: header: funny = %d\n", header.funny);
  printf("[rayt] load_ply: header: nvertices = %d\n", header.nvertices);
  printf("[rayt] load_ply: header: nvertexfields = %d\n", header.nvertexfields);
  printf("[rayt] load_ply: header: nfaces = %d\n", header.nfaces);
  printf("[rayt] load_ply: header: (x,y,z,nx,ny,nz,u,v) field positions = (%d,%d,%d,%d,%d,%d,%d,%d)\n", header.xpos, header.ypos, header.zpos, header.nxpos, header.nypos, header.nzpos, header.upos, header.vpos);

  model->arr_vertices = NULL;
  model->arr_faces = NULL;

  if (header.funny) {
    parse_vertices(&contents, &header, model);
    parse_faces(&contents, &header, model);
  }
  else {
    parse_faces(&contents, &header, model);
    parse_vertices(&contents, &header, model);
  }

  free(contents_start);
}
