#include "parser_common.h"

void free_model(Model *model) {
  arrfree(model->arr_vertices);
  arrfree(model->arr_faces);
}

char *read_whole_file(const char *filename) {
  int fd = open(filename, O_RDONLY);
  if (fd == -1) ERRORNO("open(%s)", filename);

  off_t end = lseek(fd, 0, SEEK_END);
  if (end == (off_t)-1) ERRORNO("lseek(%s)", filename);
  DEBUG("opened %s, file size = %ld", filename, (long)end);

  char *contents = malloc(end + 1);
  if (lseek(fd, 0, SEEK_SET) == (off_t)-1) ERRORNO("lseek(%s)", filename);
  if (read(fd, contents, end) == -1) ERRORNO("read(%s)", filename);
  contents[end] = '\0';
  if (close(fd) == -1) ERRORNO("close(%s)", filename);
  return contents;
}

static void skip_spaces(char **ptr) {
  while (**ptr == ' ' || **ptr == '\t') (*ptr)++;
}

void end_line(char **ptr) {
  while (**ptr != '\n') (*ptr)++;
}

void next_line(char **ptr) {
  while (**ptr != '\n') (*ptr)++;
  (*ptr)++;
}

bool consume(char **ptr, char *s) {
  char *oldptr = *ptr;
  while (*s) {
    if (*s != **ptr) goto nonmatch;
    s++; (*ptr)++;
  }
  skip_spaces(ptr);
  return true;

nonmatch:
  *ptr = oldptr;
  return false;
}

bool consume_int(char **ptr, int *i) {
  char *ptr_original = *ptr;
  int result = 0;
  int numdigits = 10;
  char digit;
  while (1) { 
    digit = **ptr; numdigits--;
    if (numdigits < 0) return false;
    if (digit < '0' || digit > '9') {
      if (digit == ' ' && digit == '\t') { skip_spaces(ptr); break; }
      else if (digit == '\n') break;
      else return false;
    }
    digit -= '0';
    result = result*10 + digit;
    (*ptr)++;
  }
  *i = result;
  return true;
}

void consume_any(char **ptr, char **s, int *len) {
  *s = *ptr;
  *len = 0;
  while (1) {
    if (**ptr == '\n') return;
    if (**ptr == ' ' && **ptr == '\t') { skip_spaces(ptr); return; }
    (*len)++; (*ptr)++;
  }
}

void swallow(char **ptr) {
  while (1) {
    if (**ptr == '\n') return;
    if (**ptr == ' ' && **ptr == '\t') { skip_spaces(ptr); return; }
    (*ptr)++;
  }
}
