#include "load_obj.h"

#define NO_MTL -1
#define LEX_STR_MAX 65

#define v3 _v3

typedef enum {
  NEWLINE, END, FLOAT, INT, STR, SLASH,

  // OBJ-specific
  VERTEX, VERTEXTEXTURE, VERTEXNORMAL, FACE, GROUP, OBJECT, SMOOTHSHADE, MTLLIB, USEMTL,

  // MTL-specific
  NEWMTL,
  KA, KD, KS, NS,
  MAP_KA, MAP_KD, MAP_KS, MAP_NS,
  NI, DISSOLVE, ILLUM
} TokenType;

typedef struct {
  int line;
  TokenType type;
  union {
    int i;
    float f;
    char *s;
  };
} Token;

typedef struct {
  bool debug;
  FILE *fp;
  const char *filename;
  int curline;
  Token *tokens; int curtoken;

  v3 *vertices, *vertextextures;
  MtlParams *mtls; int curmtl;
  Face *faces;
} ParseState;

// ------------------------------------------------------------ 
// LEXER
// ------------------------------------------------------------ 

#define lex_error(fmt, ps, ...)						\
  printf("Lexing error at %s:%d -- " fmt "\n", (ps)->filename, (ps)->curline __VA_OPT__(,) __VA_ARGS__)

#define debug(fmt, ps, ...) \
  if ((ps)->debug) printf(fmt __VA_OPT__(,) __VA_ARGS__)

char lex_advance(ParseState *ps) {
  return getc(ps->fp);
}

void lex_putback(char c, ParseState *ps) {
  ungetc(c, ps->fp);
}

char lex_peek(ParseState *ps) {
  char c = getc(ps->fp);
  lex_putback(c, ps);
  return c;
}

bool lex_matchstr(char *s, ParseState *ps) {
  int n = strlen(s);
  char peeks[n];

  for (int i = 0; i < n; i++) {
    if (lex_peek(ps) != s[i]) {
      for (int j = i-1; j >= 0; j--)
	lex_putback(peeks[j], ps);
      return false;
    }
    peeks[i] = lex_advance(ps);
  }

  return true;
}

#define lex_matchthenspace(s, ps) \
  lex_matchstr(s " ", ps) || lex_matchstr(s "\t", ps) || \
  lex_matchstr(s "\n", ps) || lex_matchstr(s "\r", ps)

void lex_addothertoken(TokenType type, ParseState *ps) {
  switch (type) {
  case VERTEX: debug("VERTEX\n", ps); break;
  case VERTEXTEXTURE: debug("VERTEXTEXTURE\n", ps); break;
  case VERTEXNORMAL: debug("VERTEXNORMAL\n", ps); break;
  case GROUP: debug("GROUP\n", ps); break;
  case FACE: debug("FACE\n", ps); break;
  case OBJECT: debug("OBJECT\n", ps); break;
  case SMOOTHSHADE: debug("SMOOTHSHADE\n", ps); break;
  case MTLLIB: debug("MTLLIB\n", ps); break;
  case USEMTL: debug("USEMTL\n", ps); break;
  case SLASH: debug("SLASH\n", ps); break;
  case NEWLINE: debug("NEWLINE\n", ps); break;

  case NEWMTL: debug("NEWMTL\n", ps); break;
  case KA: debug("KA\n", ps); break;
  case KD: debug("KD\n", ps); break;
  case KS: debug("KS\n", ps); break;
  case NS: debug("NS\n", ps); break;
  case MAP_KA: debug("MAP_KA\n", ps); break;
  case MAP_KD: debug("MAP_KD\n", ps); break;
  case MAP_KS: debug("MAP_KS\n", ps); break;
  case MAP_NS: debug("MAP_NS\n", ps); break;
  case NI: debug("NI\n", ps); break;
  case DISSOLVE: debug("DISSOLVE\n", ps); break;
  case ILLUM: debug("ILLUM\n", ps); break;

  case END: debug("END\n", ps); break;
  default: debug("???\n", ps); break;
  }
  Token tok = {.line = ps->curline, .type = type};
  arrpush(ps->tokens, tok);
}

void lex_addint(int i, ParseState *ps) {
  debug("INT %d\n", ps, i);
  Token tok = {.line = ps->curline, .type = INT, .i = i};
  arrpush(ps->tokens, tok);
}

void lex_addfloat(float f, ParseState *ps) {
  debug("FLOAT %f\n", ps, f);
  Token tok = {.line = ps->curline, .type = FLOAT, .f = f};
  arrpush(ps->tokens, tok);
}

void lex_addstr(char *s, ParseState *ps) {
  debug("STR %s\n", ps, s);
  if (strlen(s) > 64) lex_error("string has >64 characters", ps);
  Token tok = {.line = ps->curline, .type = STR, .s = s};
  arrpush(ps->tokens, tok);
}

void lex_num(ParseState *ps) {
  bool first_char = true;
  bool saw_minus = false;
  bool saw_decimal_point = false;
  char intchars[8]; int ni = 0; // Digits in integer part
  char fracchars[8]; int nf = 0; // Digits in fractional part

  while (true) {
    char c = lex_peek(ps);

    if (isdigit(c)) {
      if (ni > 8) lex_error("too many digits", ps);
      if (saw_decimal_point)
	fracchars[nf++] = c;
      else
	intchars[ni++] = c;
    }
    else if (c == '.') {
      if (saw_decimal_point) lex_error("number cannot have two decimal points", ps);
      saw_decimal_point = true;
    }
    else if (c == '-' && first_char) {
      saw_minus = true;
    }
    else
      break;

    lex_advance(ps);
    first_char = false;
  }

  int intpart = 0;
  int tenpow = 1;
  for (int i = ni-1; i >= 0; i--) {
    intpart += (intchars[i]-'0') * tenpow;
    tenpow *= 10;
  }

  if (!saw_decimal_point) {
    lex_addint(intpart * (saw_minus ? -1 : 1), ps);
    return;
  }

  int fracpart = 0;
  tenpow = 1;
  for (int i = nf-1; i >= 0; i--) {
    fracpart += (fracchars[i]-'0') * tenpow;
    tenpow *= 10;
  }

  lex_addfloat((intpart + ((float) fracpart) / tenpow) * (saw_minus ? -1 : 1), ps);
}

static inline bool _allowed_in_str(char c) {
  return isalpha(c) || isdigit(c) || c == '_' || c == '-' || c == '.';
}

void lex_str(ParseState *ps) {
  char s[LEX_STR_MAX]; int n = 0;
  while (_allowed_in_str(lex_peek(ps))) {
    if (n > LEX_STR_MAX) lex_error("string must have <= %d characters", ps, LEX_STR_MAX-1);
    s[n++] = lex_advance(ps);
  }
  s[n++] = '\0';

  char *s_heap = malloc(n*sizeof(char));
  strncpy(s_heap, s, n);
  lex_addstr(s_heap, ps);
}

void lex_scantoken_obj(ParseState *ps) {
  char c = lex_advance(ps);
  switch (c) {
  case EOF:
    lex_addothertoken(END, ps);
    return;

  case '#': {
    char c;
    while ((c = lex_advance(ps)) != '\n' && c != EOF) {};
    lex_putback(c, ps);
    break;
  }

  case 'v':
    if (isspace(lex_peek(ps))) lex_addothertoken(VERTEX, ps);
    else if (lex_matchthenspace("t", ps)) lex_addothertoken(VERTEXTEXTURE, ps);
    else if (lex_matchthenspace("n", ps)) lex_addothertoken(VERTEXNORMAL, ps);
    else { lex_putback('v', ps); lex_str(ps); }
    break;

  case 'f':
    if (isspace(lex_peek(ps))) lex_addothertoken(FACE, ps);
    else { lex_putback('f', ps); lex_str(ps); }
    break;

  case 'g':
    if (isspace(lex_peek(ps))) lex_addothertoken(GROUP, ps);
    else { lex_putback('g', ps); lex_str(ps); }
    break;

  case 'o':
    if (isspace(lex_peek(ps))) lex_addothertoken(OBJECT, ps);
    else { lex_putback('o', ps); lex_str(ps); }
    break;

  case 's':
    if (isspace(lex_peek(ps))) lex_addothertoken(SMOOTHSHADE, ps);
    else { lex_putback('s', ps); lex_str(ps); }
    break;

  case 'm':
    if (lex_matchthenspace("tllib", ps)) lex_addothertoken(MTLLIB, ps);
    else { lex_putback('m', ps); lex_str(ps); }
    break;

  case 'u':
    if (lex_matchthenspace("semtl", ps)) lex_addothertoken(USEMTL, ps);
    else { lex_putback('u', ps); lex_str(ps); }
    break;

  case '\n':
    ps->curline++; lex_addothertoken(NEWLINE, ps); break;

  case '/':
    lex_addothertoken(SLASH, ps); break;

  case ' ': case '\r': case '\t':
    break;

  default:
    if (isdigit(c) || c == '-') {
      lex_putback(c, ps);
      lex_num(ps);
    }
    else if (_allowed_in_str(c)) {
      lex_putback(c, ps);
      lex_str(ps);
    }
    else
      lex_error("unexpected character %c", ps, c);
  }
}

void lex_scantoken_mtl(ParseState *ps) {
  char c = lex_advance(ps);
  switch (c) {
  case EOF:
    lex_addothertoken(END, ps);
    lex_advance(ps);
    return;

  case '#': {
    char c;
    while ((c = lex_advance(ps)) != '\n' && c != EOF) {};
    lex_putback(c, ps);
    break;
  }

  case 'n':
    if (lex_matchthenspace("ewmtl", ps)) lex_addothertoken(NEWMTL, ps);
    else { lex_putback('n', ps); lex_str(ps); }
    break;

  case 'N':
    if (lex_matchthenspace("s", ps)) lex_addothertoken(NS, ps);
    else if (lex_matchthenspace("i", ps)) lex_addothertoken(NI, ps);
    else { lex_putback('N', ps); lex_str(ps); }
    break;

  case 'm':
    if (lex_matchthenspace("ap_Ka", ps)) lex_addothertoken(MAP_KA, ps);
    else if (lex_matchthenspace("ap_Kd", ps)) lex_addothertoken(MAP_KD, ps);
    else if (lex_matchthenspace("ap_Ks", ps)) lex_addothertoken(MAP_KS, ps);
    else if (lex_matchthenspace("ap_Ns", ps)) lex_addothertoken(MAP_NS, ps);
    else { lex_putback('m', ps); lex_str(ps); }
    break;

  case 'i':
    if (lex_matchthenspace("llum", ps)) lex_addothertoken(ILLUM, ps);
    else { lex_putback('i', ps); lex_str(ps); }
    break;

  case 'K':
    if (lex_matchthenspace("a", ps)) lex_addothertoken(KA, ps);
    else if (lex_matchthenspace("d", ps)) lex_addothertoken(KD, ps);
    else if (lex_matchthenspace("s", ps)) lex_addothertoken(KS, ps);
    else { lex_putback('K', ps); lex_str(ps); }
    break;

  case '\n':
    ps->curline++; lex_addothertoken(NEWLINE, ps); break;

  case ' ': case '\r': case '\t':
    break;

  default:
    if (isdigit(c) || c == '-') {
      lex_putback(c, ps);
      lex_num(ps);
    }
    else if (_allowed_in_str(c)) {
      lex_putback(c, ps);
      lex_str(ps);
    }
    else
      lex_error("unexpected character %c", ps, c);
  }
}

// ------------------------------------------------------------ 
// PARSER
// ------------------------------------------------------------ 

#define parse_error(fmt, ps, ...)					\
  { printf("Parsing error at %s:%d -- " fmt "\n", (ps)->filename, (ps)->tokens[ps->curtoken].line __VA_OPT__(,) __VA_ARGS__); exit(1); }

Token parse_advance(ParseState *ps) {
  return ps->tokens[ps->curtoken++];
}

Token parse_peekprev(ParseState *ps) {
  return ps->tokens[ps->curtoken-1];
}

bool parse_matchtoken(TokenType type, ParseState *ps) {
  if (ps->tokens[ps->curtoken].type == type) {
    ps->curtoken++;
    return true;
  }
  return false;
}

bool parse_newlineorend(ParseState *ps) {
  return parse_matchtoken(NEWLINE, ps) || parse_matchtoken(END, ps);
}

v3 parse_two_floats(ParseState *ps) {
  float x, y;
  if (parse_matchtoken(INT, ps)) x = parse_peekprev(ps).i;
  else if (parse_matchtoken(FLOAT, ps)) x = parse_peekprev(ps).f;
  else parse_error("expected a number", ps);
  debug("x = %f\n", ps, x);

  if (parse_matchtoken(INT, ps)) y = parse_peekprev(ps).i;
  else if (parse_matchtoken(FLOAT, ps)) y = parse_peekprev(ps).f;
  else parse_error("expected a number", ps);
  debug("y = %f\n", ps, y);

  v3 v = {x, y, 0};

  if (parse_newlineorend(ps)) return v;

  // For .obj vertex textures, there is an optional w coordinate which we will ignore
  if (parse_matchtoken(INT, ps) || parse_matchtoken(FLOAT, ps)) {}
  else parse_error("expected a number", ps);

  if (!parse_newlineorend(ps)) parse_error("too many numbers", ps);

  return v;
}

v3 parse_three_floats(ParseState *ps) {
  float x, y, z;
  if (parse_matchtoken(INT, ps)) x = parse_peekprev(ps).i;
  else if (parse_matchtoken(FLOAT, ps)) x = parse_peekprev(ps).f;
  else parse_error("expected a number", ps);
  debug("x = %f\n", ps, x);

  if (parse_matchtoken(INT, ps)) y = parse_peekprev(ps).i;
  else if (parse_matchtoken(FLOAT, ps)) y = parse_peekprev(ps).f;
  else parse_error("expected a number", ps);
  debug("y = %f\n", ps, y);

  if (parse_matchtoken(INT, ps)) z = parse_peekprev(ps).i;
  else if (parse_matchtoken(FLOAT, ps)) z = parse_peekprev(ps).f;
  else parse_error("expected a number", ps);
  debug("z = %f\n", ps, z);

  v3 v = {x, y, z};

  if (parse_newlineorend(ps)) return v;

  // For .obj vertices, there is an optional w coordinate which we will ignore
  if (parse_matchtoken(INT, ps) || parse_matchtoken(FLOAT, ps)) {}
  else parse_error("expected a number", ps);

  if (!parse_newlineorend(ps)) parse_error("too many numbers", ps);

  return v;
}

void parse_vertex(ParseState *ps) {
  debug("PARSING VERTEX\n", ps);
  arrpush(ps->vertices, parse_three_floats(ps));
}

void parse_vertextexture(ParseState *ps) {
  debug("PARSING VERTEXTEXTURE\n", ps);
  arrpush(ps->vertextextures, parse_two_floats(ps));
}

void parse_vertexnormal(ParseState *ps) {
  debug("PARSING VERTEXNORMAL\n", ps);
  while (parse_advance(ps).type != NEWLINE) {};
}

static inline int _get_absolute_idx(int idx, ParseState *ps) {
  if (idx < 0) {
    if (arrlen(ps->vertices)+idx < 0)
      parse_error("negative vertex index is out of bounds", ps);
    return arrlen(ps->vertices) + idx;
  }
  else {
    idx--;  // vertices in .objs are 1-indexed
    if (idx >= arrlen(ps->vertices))
      parse_error("positive vertex index is out of bounds", ps);
    return idx;
  }
}

void parse_face(ParseState *ps) {
  debug("PARSING FACE\n", ps);
  v3 vs[10], vts[10]; int k = 0;

  while (true) {
    if (k >= 10) parse_error("maximum 10 vertices allowed in face", ps);

    // Vertex index
    if (parse_matchtoken(INT, ps)) {
      int idx = _get_absolute_idx(parse_peekprev(ps).i, ps);
      debug("face %d: vertex %d\n", ps, k+1, idx);
      vs[k++] = ps->vertices[idx];
    }
    else if (parse_newlineorend(ps)) break;
    else parse_error("vertex index should be an integer", ps);

    // Go to next entry if no /
    if (!parse_matchtoken(SLASH, ps)) continue;

    // Texture index
    if (parse_matchtoken(INT, ps)) {
      int idx = _get_absolute_idx(parse_peekprev(ps).i, ps);
      debug("face %d: vertextexture %d\n", ps, k+1, idx);
      vts[k++] = ps->vertextextures[idx];
    }
    else if (parse_newlineorend(ps)) break;
    else if (!parse_matchtoken(SLASH, ps)) parse_error("expected integer after f command", ps);

    // Go to next entry if no /
    if (!parse_matchtoken(SLASH, ps)) continue;

    // Normal index
    if (parse_matchtoken(INT, ps)) {}
    else if (parse_newlineorend(ps)) break;
    else if (!parse_matchtoken(SLASH, ps)) parse_error("expected integer after f command", ps);

    if (parse_matchtoken(SLASH, ps)) parse_error("each face entry has at most 3 fields", ps);
  }

  if (k < 3) parse_error("face must have at least 3 vertices", ps);

  for (int i = 1; i < k-1; i++) {
    Face face;
    face.vs[0] = vs[0];
    face.vs[1] = vs[i];
    face.vs[2] = vs[i+1];
    face.vts[0] = vts[0];
    face.vts[1] = vts[i];
    face.vts[2] = vts[i+1];
    face.mtl = ps->mtls[ps->curmtl];
    arrpush(ps->faces, face);
  }
}

void parse_group(ParseState *ps) {
  debug("PARSING GROUP\n", ps);
  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_object(ParseState *ps) {
  debug("PARSING OBJECT\n", ps);
  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_smoothshade(ParseState *ps) {
  debug("PARSING SMOOTHSHADE\n", ps);
  while (parse_advance(ps).type != NEWLINE) {};
}

// Defined later
void read_mtl(const char *filename, ParseState *ps);

void parse_mtllib(ParseState *ps) {
  debug("PARSING MTLLIB\n", ps);
  Token tok = ps->tokens[ps->curtoken];
  if (tok.type != STR) parse_error("expected string", ps);
  read_mtl(tok.s, ps);
  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_usemtl(ParseState *ps) {
  debug("PARSING USEMTL\n", ps);
  Token tok = parse_advance(ps);
  if (tok.type != STR) parse_error("expected string", ps);

  bool found_mtl = false;
  for (int i = 0; i < arrlen(ps->mtls); i++) {
    if (strcmp(ps->mtls[i].name, tok.s) == 0) {
      ps->curmtl = i;
      found_mtl = true;
      break;
    }
  }
  if (!found_mtl) parse_error("material \"%s\" could not be found", ps, tok.s);

  while (parse_advance(ps).type != NEWLINE) {};
}

Face *parse_obj(ParseState *ps) {
  while (ps->curtoken < arrlen(ps->tokens) && ps->tokens[ps->curtoken].type != END) {
    if (parse_matchtoken(VERTEX, ps)) parse_vertex(ps);
    else if (parse_matchtoken(VERTEXTEXTURE, ps)) parse_vertextexture(ps);
    else if (parse_matchtoken(VERTEXNORMAL, ps)) parse_vertexnormal(ps);
    else if (parse_matchtoken(FACE, ps)) parse_face(ps);
    else if (parse_matchtoken(GROUP, ps)) parse_group(ps);
    else if (parse_matchtoken(OBJECT, ps)) parse_object(ps);
    else if (parse_matchtoken(SMOOTHSHADE, ps)) parse_smoothshade(ps);
    else if (parse_matchtoken(MTLLIB, ps)) parse_mtllib(ps);
    else if (parse_matchtoken(USEMTL, ps)) parse_usemtl(ps);
    else parse_advance(ps);
  }
  return ps->faces;
}

void parse_newmtl(MtlParams *mtl, ParseState *ps) {
  debug("PARSING NEWMTL\n", ps);
  Token tok = parse_advance(ps);
  if (tok.type != STR) parse_error("expected string", ps);
  mtl->name = tok.s;
  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_Kx(MtlParams *mtl, char type, ParseState *ps) {
  assert(type == 'a' || type == 'd' || type == 's');
  debug("PARSING K%c\n", ps, type);
  v3 v = parse_three_floats(ps);
  if (type == 'a') mtl->Ka = v;
  else if (type == 'd') mtl->Kd = v;
  else if (type == 's') mtl->Ks = v;
}

void parse_map(MtlParams *mtl, char type, ParseState *ps) {
  assert(type == 'a' || type == 'd' || type == 's' || type == 'N');
  debug("PARSING map_K%c\n", ps, type);
  Token tok = parse_advance(ps);
  if (tok.type != STR) parse_error("expected string", ps);

  if (type == 'a') mtl->map_Ka = tok.s;
  else if (type == 'd') mtl->map_Kd = tok.s;
  else if (type == 's') mtl->map_Ks = tok.s;
  else if (type == 'N') mtl->map_Ns = tok.s;

  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_Nx(MtlParams *mtl, char type, ParseState *ps) {
  assert(type == 's' || type == 'i');
  debug("PARSING N%c\n", ps, type);
  Token tok = parse_advance(ps);
  if (tok.type != FLOAT && tok.type != INT) parse_error("expected float", ps);

  float x;
  if (tok.type == FLOAT) x = tok.f;
  else if (tok.type == INT) x = tok.i;
  if (type == 's') mtl->Ns = x;
  else if (type == 'i') mtl->Ni = x;

  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_d(MtlParams *mtl, ParseState *ps) {
  debug("PARSING D\n", ps);
  Token tok = parse_advance(ps);
  if (tok.type != FLOAT && tok.type != INT) parse_error("expected float", ps);
  if (tok.type == FLOAT) mtl->d = tok.f;
  else if (tok.type == INT) mtl->d = tok.i;
  while (parse_advance(ps).type != NEWLINE) {};
}

void parse_illum(MtlParams *mtl, ParseState *ps) {
  debug("PARSING ILLUM\n", ps);
  Token tok = parse_advance(ps);
  if (tok.type != INT) parse_error("expected int", ps);
  if (tok.i < 0 || tok.i > 10) parse_error("invalid illum number", ps);
  mtl->illum = tok.i;
  while (parse_advance(ps).type != NEWLINE) {};
}

MtlParams *parse_mtl(ParseState *ps) {
  MtlParams *mtls = NULL; int n = -1;
  MtlParams empty_mtl; memset(&empty_mtl, 0, sizeof(empty_mtl));

  while (ps->curtoken < arrlen(ps->tokens) && ps->tokens[ps->curtoken].type != END) {
    if (parse_matchtoken(NEWMTL, ps)) {
      arrpush(mtls, empty_mtl);
      n++;
      parse_newmtl(mtls+n, ps);
    }
    else if (parse_matchtoken(KA, ps)) parse_Kx(mtls+n, 'a', ps);
    else if (parse_matchtoken(KD, ps)) parse_Kx(mtls+n, 'd', ps);
    else if (parse_matchtoken(KS, ps)) parse_Kx(mtls+n, 's', ps);
    else if (parse_matchtoken(MAP_KA, ps)) parse_map(mtls+n, 'a', ps);
    else if (parse_matchtoken(MAP_KD, ps)) parse_map(mtls+n, 'd', ps);
    else if (parse_matchtoken(MAP_KS, ps)) parse_map(mtls+n, 's', ps);
    else if (parse_matchtoken(MAP_NS, ps)) parse_map(mtls+n, 'N', ps);
    else if (parse_matchtoken(NS, ps)) parse_Nx(mtls+n, 's', ps);
    else if (parse_matchtoken(NI, ps)) parse_Nx(mtls+n, 'i', ps);
    else if (parse_matchtoken(DISSOLVE, ps)) parse_d(mtls+n, ps);
    else if (parse_matchtoken(ILLUM, ps)) parse_illum(mtls+n, ps);
    else parse_advance(ps);
  }
  return mtls;
}

void read_mtl(const char *filename, ParseState *ps) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    char s[100];  // TODO don't hardcode maximum array size
    parse_error("could not open %s (%s)", ps, filename, s);
  }

  ParseState mtl_ps;
  memset(&mtl_ps, 0, sizeof(mtl_ps));
  mtl_ps.filename = filename;
  mtl_ps.fp = fp;
  mtl_ps.debug = true;
  while (!feof(mtl_ps.fp))
    lex_scantoken_mtl(&mtl_ps);

  MtlParams *mtls = parse_mtl(&mtl_ps);
  for (int i = 0; i < arrlen(mtls); i++) {
    arrpush(ps->mtls, mtls[i]);
  }
  arrfree(mtls);
  arrfree(mtl_ps.tokens);
}

Face *load_obj(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    char s[100];  // TODO don't hardcode maximum array size
    printf("Error -- could not open %s", filename);
    perror(s);
    exit(1);
  }

  ParseState ps;
  memset(&ps, 0, sizeof(ps));
  ps.filename = filename;
  ps.fp = fp;
  ps.curmtl = NO_MTL;

  while (!feof(ps.fp))
    lex_scantoken_obj(&ps);

  Face *faces = parse_obj(&ps);
  ps.debug = true;
  debug("%d faces parsed\n", &ps, (int)arrlen(faces));
  for (int i = 0; i < arrlen(faces); i++) {
    Face f = faces[i];
    debug("face %d:\n", &ps, i);
    debug("v1 = %f %f %f\n", &ps, f.vs[0].x, f.vs[0].y, f.vs[0].z);
    debug("v2 = %f %f %f\n", &ps, f.vs[1].x, f.vs[1].y, f.vs[1].z);
    debug("v3 = %f %f %f\n", &ps, f.vs[2].x, f.vs[2].y, f.vs[2].z);
    debug("vt1 = %f %f %f\n", &ps, f.vts[0].x, f.vts[0].y, f.vts[0].z);
    debug("vt2 = %f %f %f\n", &ps, f.vts[1].x, f.vts[1].y, f.vts[1].z);
    debug("vt3 = %f %f %f\n", &ps, f.vts[2].x, f.vts[2].y, f.vts[2].z);
    debug("Ka = %f %f %f\n", &ps, f.mtl.Ka.x, f.mtl.Ka.y, f.mtl.Ka.z);
    debug("Kd = %f %f %f\n", &ps, f.mtl.Kd.x, f.mtl.Kd.y, f.mtl.Kd.z);
    debug("Ks = %f %f %f\n", &ps, f.mtl.Ks.x, f.mtl.Ks.y, f.mtl.Ks.z);
    debug("map_Ka = %s, map_Kd = %s, map_Ks = %s\n", &ps, f.mtl.map_Ka, f.mtl.map_Kd, f.mtl.map_Ks);
    debug("Ns = %f, Ni = %f, d = %f\n", &ps, f.mtl.Ns, f.mtl.Ni, f.mtl.d);
    debug("illum = %d\n\n", &ps, f.mtl.illum);
  }

  arrfree(ps.tokens);
  arrfree(ps.vertices);
  arrfree(ps.vertextextures);
  arrfree(ps.mtls);
  return faces;
}

#undef v3