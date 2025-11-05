// wfagpu_pyshim.c  (fixed: only uses public fields/functions)
// Build:
//   gcc -fPIC -shared wfagpu_pyshim.c -o libwfagpu_pyshim.so \
//     -I $WFAGPU_PATH/lib/ -I $WFAGPU_PATH \
//     -L $WFAGPU_PATH/build/ -L $WFAGPU_PATH/external/WFA2-lib/lib/ \
//     -lwfagpu -lwfa -lm -fopenmp
// Runtime:
//   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WFAGPU_PATH/build"
//# Build the project (creates libwfagpu.so and the CLI tool)
//git clone https://github.com/quim0/WFA-GPU.git
//cd WFA-GPU
//git submodule update --init --recursive
//./build.sh
//export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/build"

//# Build the shim
//gcc -fPIC -shared wfagpu_pyshim.c -o libwfagpu_pyshim.so \
//  -I ./lib/ -I . \
//  -L ./build/ -L ./external/WFA2-lib/lib/ \
//  -lwfagpu -lwfa -lm -fopenmp

//# Python smoke test (exact via shim)
//python3 - <<'PY'
//from wfagpu import wfa_gpu
//#w = WFAGPU("./libwfagpu_pyshim.so", "./bin/wfa.affine.gpu")
//print("LIB:", w.align("GAATA","GATACA", x=2,o=3,e=1, compute_cigar=True, batch_size=10))
//# Now ask for banding -> switches to CLI backend automatically
//print("CLI:", w.align("CATTAATCTT","CAGTAAT", x=2,o=3,e=1, compute_cigar=True, band="auto", max_distance=3000, threads_per_block=512))
//PY

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "include/wfa_gpu.h"

static char* json_escape(const char* s){
  if(!s) return NULL;
  size_t n=0; for(const char* p=s; *p; ++p) n += (*p=='"'||*p=='\\')?2:1;
  char* out=(char*)malloc(n+1); char* w=out;
  for(const char* p=s; *p; ++p){ if(*p=='"'||*p=='\\'){*w++='\\';*w++=*p;} else {*w++=*p;} }
  *w='\0'; return out;
}

const char* wfagpu_align_pair_json(
  const char* query, const char* target,
  int x, int o, int e,
  int compute_cigar,      // 0/1
  int batch_size          // <=0 -> default
){
  wfagpu_aligner_t aligner = (wfagpu_aligner_t){0};
  wfagpu_initialize_aligner(&aligner);

  // sequences
  wfagpu_add_sequences(&aligner, query, target);

  // penalties
  affine_penalties_t penalties = {.x=x, .o=o, .e=e};
  wfagpu_initialize_parameters(&aligner, penalties);

  if(batch_size>0) wfagpu_set_batch_size(&aligner, batch_size);

  // only documented public field
  aligner.alignment_options.compute_cigar = (compute_cigar!=0);

  // run
  wfagpu_align(&aligner);

  int error = aligner.results[0].error;
  const char* cigar = (aligner.alignment_options.compute_cigar && aligner.results[0].cigar.buffer)
                        ? aligner.results[0].cigar.buffer : NULL;

  char* cigar_esc = cigar ? json_escape(cigar) : NULL;
  size_t cap = 256 + (cigar_esc?strlen(cigar_esc):0);
  char* js = (char*)malloc(cap);
  if(!js){ const char* oom="{\"ok\":false,\"error\":\"oom\"}";
          char* r=(char*)malloc(strlen(oom)+1); if(r) strcpy(r,oom); return r; }

  if(cigar){
    snprintf(js, cap,
      "{\"ok\":true,\"score\":%d,\"error\":%d,\"cigar\":\"%s\",\"query_len\":%zu,\"target_len\":%zu}",
      error,error,cigar_esc,strlen(query),strlen(target));
  }else{
    snprintf(js, cap,
      "{\"ok\":true,\"score\":%d,\"error\":%d,\"cigar\":null,\"query_len\":%zu,\"target_len\":%zu}",
      error,error,strlen(query),strlen(target));
  }
  if(cigar_esc) free(cigar_esc);
  return js;
}

void wfagpu_free_string(const char* s){ if(s) free((void*)s); }
