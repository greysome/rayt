#ifndef _LOAD_PLY_H
#define _LOAD_PLY_H

#include "parser_common.h"

#ifdef __CUDACC__
extern "C" void load_ply(char *filename, Model *model);
#else
void load_ply(char *filename, Model *model);
#endif

#endif
