#ifndef _LOAD_OBJ_H
#define _LOAD_OBJ_H

#include "parser_common.h"

#ifdef __CUDACC__
extern "C" Face *load_obj(const char *filename);
#else
Face *load_obj(const char *filename);
#endif

#endif
