// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "ptx_parser.h"
#include <regex>
#include "../../libcuda/gpgpu_context.h"
#include "ptx_ir.h"

typedef void *yyscan_t;
#include <stdarg.h>
#include "ptx.tab.h"

extern int ptx_get_lineno(yyscan_t yyscanner);
extern YYSTYPE *ptx_get_lval(yyscan_t yyscanner);
extern int ptx_error(yyscan_t yyscanner, ptx_recognizer *recognizer,
                     const char *s);
extern int ptx_lex_init(yyscan_t *scanner);
extern void ptx_set_in(FILE *_in_str, yyscan_t yyscanner);
extern FILE *ptx_get_in(yyscan_t yyscanner);
extern int ptx_parse(yyscan_t scanner, ptx_recognizer *recognizer);
extern int ptx_lex_destroy(yyscan_t scanner);

void ptx_recognizer::set_ptx_warp_size(const struct core_config *warp_size) {
  g_shader_core_config = warp_size;
}

#define PTX_PARSE_DPRINTF(...)                                            \
  if (g_debug_ir_generation) {                                            \
    printf(" %s:%u => ", gpgpu_ctx->g_filename, ptx_get_lineno(scanner)); \
    printf("   (%s:%u) ", __FILE__, __LINE__);                            \
    printf(__VA_ARGS__);                                                  \
    printf("\n");                                                         \
    fflush(stdout);                                                       \
  }

static std::map<unsigned, std::string> g_ptx_token_decode;

const char *decode_token(int type) { return g_ptx_token_decode[type].c_str(); }

void ptx_recognizer::read_parser_environment_variables() {
  gpgpu_ctx->g_filename = getenv("PTX_SIM_KERNELFILE");
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    int debug_execution = 0;
    sscanf(dbg_level, "%d", &debug_execution);
    if (debug_execution >= 30) g_debug_ir_generation = true;
  }
}

void ptx_recognizer::init_directive_state() {
  PTX_PARSE_DPRINTF("init_directive_state");
  g_space_spec = undefined_space;
  g_ptr_spec = undefined_space;
  g_scalar_type_spec = -1;
  g_vector_spec = -1;
  g_opcode = -1;
  g_alignment_spec = -1;
  g_size = 0;
  g_extern_spec = 0;
  g_scalar_type.clear();
  g_operands.clear();
  g_last_symbol = NULL;
}

void ptx_recognizer::init_instruction_state() {
  PTX_PARSE_DPRINTF("init_instruction_state");
  g_pred = NULL;
  g_neg_pred = 0;
  g_pred_mod = -1;
  g_label = NULL;
  g_opcode = -1;
  g_options.clear();
  g_wmma_options.clear();
  g_vector_operands.clear();
  g_matrix_opcode.clear();
  g_mbarrier_opcode.clear();
  g_tma_opcode.clear();
  g_cluster_opcode.clear();
  g_return_var = operand_info(gpgpu_ctx);
  init_directive_state();
}

symbol_table *gpgpu_context::init_parser(const char *ptx_filename) {
  g_filename = strdup(ptx_filename);
  if (g_global_allfiles_symbol_table == NULL) {
    g_global_allfiles_symbol_table =
        new symbol_table("global_allfiles", 0, NULL, this);
    ptx_parser->g_global_symbol_table = ptx_parser->g_current_symbol_table =
        g_global_allfiles_symbol_table;
  }
  /*else {
      g_global_symbol_table = g_current_symbol_table = new
  symbol_table("global",0,g_global_allfiles_symbol_table);
  }*/

#define DEF(X, Y) g_ptx_token_decode[X] = Y;
#include "ptx_parser_decode.def"
#undef DEF
  g_ptx_token_decode[undefined_space] = "undefined_space";
  g_ptx_token_decode[undefined_space] = "undefined_space=0";
  g_ptx_token_decode[reg_space] = "reg_space";
  g_ptx_token_decode[local_space] = "local_space";
  g_ptx_token_decode[shared_space] = "shared_space";
  g_ptx_token_decode[param_space_unclassified] = "param_space_unclassified";
  g_ptx_token_decode[param_space_kernel] = "param_space_kernel";
  g_ptx_token_decode[param_space_local] = "param_space_local";
  g_ptx_token_decode[const_space] = "const_space";
  g_ptx_token_decode[tex_space] = "tex_space";
  g_ptx_token_decode[surf_space] = "surf_space";
  g_ptx_token_decode[global_space] = "global_space";
  g_ptx_token_decode[generic_space] = "generic_space";
  g_ptx_token_decode[instruction_space] = "instruction_space";

  ptx_lex_init(&(ptx_parser->scanner));
  ptx_parser->init_directive_state();
  ptx_parser->init_instruction_state();

  FILE *ptx_in;
  ptx_in = fopen(ptx_filename, "r");
  ptx_set_in(ptx_in, ptx_parser->scanner);
  const int parse_result = ptx_parse(ptx_parser->scanner, ptx_parser);
  ptx_in = ptx_get_in(ptx_parser->scanner);
  ptx_lex_destroy(ptx_parser->scanner);
  fclose(ptx_in);
  if (parse_result != 0 || ptx_parser->g_error_detected) {
    fprintf(stderr, "%s: PTX parsing failed; refusing partial module\n", ptx_filename);
    exit(1);
  }
  return ptx_parser->g_global_symbol_table;
}

void ptx_recognizer::start_function(int entry_point) {
  PTX_PARSE_DPRINTF("start_function");
  init_directive_state();
  init_instruction_state();
  g_entry_point = entry_point;
  g_func_info = NULL;
  g_entry_func_param_index = 0;
}

void ptx_recognizer::add_function_name(const char *name) {
  PTX_PARSE_DPRINTF(
      "add_function_name %s %s", name,
      ((g_entry_point == 1) ? "(entrypoint)"
                            : ((g_entry_point == 2) ? "(extern)" : "")));
  bool prior_decl = g_global_symbol_table->add_function_decl(
      name, g_entry_point, &g_func_info, &g_current_symbol_table);
  if (g_add_identifier_cached__identifier) {
    add_identifier(g_add_identifier_cached__identifier,
                   g_add_identifier_cached__array_dim,
                   g_add_identifier_cached__array_ident);
    free(g_add_identifier_cached__identifier);
    g_add_identifier_cached__identifier = NULL;
    g_func_info->add_return_var(g_last_symbol);
    init_directive_state();
  }
  if (prior_decl) {
    g_func_info->remove_args();
  }
  g_global_symbol_table->add_function(g_func_info, gpgpu_ctx->g_filename,
                                      ptx_get_lineno(scanner));
}

// Jin: handle instruction group for cdp
void ptx_recognizer::start_inst_group() {
  PTX_PARSE_DPRINTF("start_instruction_group");
  g_current_symbol_table = g_current_symbol_table->start_inst_group();
}

void ptx_recognizer::end_inst_group() {
  PTX_PARSE_DPRINTF("end_instruction_group");
  g_current_symbol_table = g_current_symbol_table->end_inst_group();
}

void ptx_recognizer::add_directive() {
  PTX_PARSE_DPRINTF("add_directive");
  init_directive_state();
}

#define mymax(a, b) ((a) > (b) ? (a) : (b))

void ptx_recognizer::end_function() {
  PTX_PARSE_DPRINTF("end_function");

  init_directive_state();
  init_instruction_state();
  g_max_regs_per_thread = mymax(g_max_regs_per_thread,
                                (g_current_symbol_table->next_reg_num() - 1));
  g_func_info->add_inst(g_instructions);
  g_instructions.clear();
  gpgpu_ptx_assemble(g_func_info->get_name(), g_func_info);
  g_current_symbol_table = g_global_symbol_table;

  PTX_PARSE_DPRINTF("function %s, PC = %llu\n", g_func_info->get_name().c_str(),
                    g_func_info->get_start_PC());
}

#define parse_error(msg, ...) \
  parse_error_impl(__FILE__, __LINE__, msg, ##__VA_ARGS__)
#define parse_assert(cond, msg, ...) \
  parse_assert_impl((cond), __FILE__, __LINE__, msg, ##__VA_ARGS__)

void ptx_recognizer::parse_error_impl(const char *file, unsigned line,
                                      const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  g_error_detected = 1;
  printf("%s:%u: Parse error: %s (%s:%u)\n\n", gpgpu_ctx->g_filename,
         ptx_get_lineno(scanner), buf, file, line);
  ptx_error(scanner, this, NULL);
  abort();
  exit(1);
}

void ptx_recognizer::parse_assert_impl(int test_value, const char *file,
                                       unsigned line, const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  if (test_value == 0) parse_error_impl(file, line, "%s", buf);
}

void ptx_recognizer::set_return() {
  parse_assert((g_opcode == CALL_OP || g_opcode == CALLP_OP),
               "only call can have return value");
  g_operands.front().set_return();
  g_return_var = g_operands.front();
}

const ptx_instruction *ptx_recognizer::ptx_instruction_lookup(
    const char *filename, unsigned linenumber) {
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >::iterator
      f = g_inst_lookup.find(filename);
  if (f == g_inst_lookup.end()) return NULL;
  std::map<unsigned, const ptx_instruction *>::iterator l =
      f->second.find(linenumber);
  if (l == f->second.end()) return NULL;
  return l->second;
}

void ptx_recognizer::add_instruction() {
  validate_matrix_operands();
  validate_mbarrier_operands();
  validate_tma_operands();
  if (g_opcode == MAPA_OP || g_opcode == GETCTARANK_OP) {
    std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
    parse_assert(ops.size() == (g_opcode == MAPA_OP ? 3 : 2),
                 "cluster address operand count mismatch");
    const unsigned type = g_scalar_type.front();
    const auto width = [](const operand_info &op, unsigned bits) {
      if (!op.is_reg()) return false;
      size_t size; int sign;
      type_info_key::type_decode(op.get_symbol()->type()->get_key().scalar_type(), size, sign);
      return size == bits;
    };
    parse_assert(width(ops[0], g_opcode == GETCTARANK_OP ? 32 : type == U64_TYPE ? 64 : 32) &&
                 width(ops[1], type == U64_TYPE ? 64 : 32),
                 "cluster address requires registers of the declared width");
    if (g_opcode == MAPA_OP)
      parse_assert(width(ops[2], 32), "mapa CTA rank requires a 32-bit register");
  }
  if (g_opcode == SETMAXNREG_OP) {
    parse_assert(g_operands.size() == 1 &&
                 (g_operands.front().get_type() == int_t || g_operands.front().get_type() == unsigned_t),
                 "setmaxnreg requires one integer immediate register count");
    const uint64_t count = g_operands.front().get_literal_value().u64;
    parse_assert(count >= 24 && count <= 256 && count % 8 == 0,
                 "setmaxnreg count must be 24..256 and a multiple of 8");
  }
  if (g_opcode == ELECT_OP) {
    std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
    parse_assert(ops.size() == 2 && ops[0].is_vector() &&
                 ops[0].get_vect_nelem() == 2 &&
                 ops[0].get_double_operand_type() == -2,
                 "elect.sync requires a 32-bit register|predicate destination and membermask");
    const symbol *lane = ops[0].vec_symbol(0), *pred = ops[0].vec_symbol(1);
    parse_assert(lane->is_reg() && pred->is_reg() &&
                 (lane->type()->get_key().scalar_type() == B32_TYPE ||
                  lane->type()->get_key().scalar_type() == U32_TYPE) &&
                 pred->type()->get_key().scalar_type() == PRED_TYPE,
                 "elect.sync destination types must be .b32/.u32 and .pred");
    const bool register_mask = ops[1].is_reg() && !ops[1].is_vector() &&
        (ops[1].get_symbol()->type()->get_key().scalar_type() == B32_TYPE ||
         ops[1].get_symbol()->type()->get_key().scalar_type() == U32_TYPE);
    parse_assert(register_mask || ops[1].get_type() == int_t ||
                 ops[1].get_type() == unsigned_t,
                 "elect.sync membermask must be a 32-bit register or integer immediate");
  }
  if (g_opcode == PREFETCH_TENSORMAP_OP) {
    parse_assert(g_operands.size() == 1 && g_operands.front().is_memory_operand(),
                 "prefetch.tensormap requires one bracketed address");
    const symbol *address = g_operands.front().get_symbol();
    parse_assert(address && (!address->is_reg() ||
                 address->type()->get_key().scalar_type() == B64_TYPE ||
                 address->type()->get_key().scalar_type() == U64_TYPE),
                 "prefetch.tensormap address register must be 64-bit");
  }
  if (g_opcode == HOPPER_FRONTEND_ONLY_OP) {
    if (g_cluster_opcode == "griddepcontrol.wait") {
      parse_assert(g_operands.empty(), "griddepcontrol.wait takes no operands");
    } else if (g_cluster_opcode == "cp.async.bulk.commit_group") {
      parse_assert(g_operands.empty(), "cp.async.bulk.commit_group takes no operands");
    } else if (g_cluster_opcode == "cp.async.bulk.wait_group.read") {
      parse_assert(g_operands.size() == 1 &&
                   (g_operands.front().get_type() == int_t ||
                    g_operands.front().get_type() == unsigned_t) &&
                   g_operands.front().get_literal_value().u64 == 0,
                   "frontend scope supports cp.async.bulk.wait_group.read immediate zero only");
    } else if (g_cluster_opcode.find("cp.async.bulk.tensor.") == 0) {
      std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
      parse_assert(ops.size() == 3 && ops[0].is_reg() && !ops[0].is_vector() &&
                   (ops[0].get_symbol()->type()->get_key().scalar_type() == B64_TYPE ||
                    ops[0].get_symbol()->type()->get_key().scalar_type() == U64_TYPE) &&
                   ops[1].is_vector() && ops[1].get_vect_nelem() == 5 &&
                   ops[2].is_memory_operand(),
                   "TMA store requires a 64-bit descriptor, five coordinates and shared address");
      const symbol *address = ops[2].get_symbol();
      parse_assert(address && (address->is_shared() ||
                   (address->is_reg() &&
                    (address->type()->get_key().scalar_type() == B32_TYPE ||
                     address->type()->get_key().scalar_type() == U32_TYPE ||
                     address->type()->get_key().scalar_type() == B64_TYPE ||
                     address->type()->get_key().scalar_type() == U64_TYPE))),
                   "TMA store shared address must be a shared symbol or 32/64-bit register");
      for (unsigned r = 0; r < 5; ++r) {
        const symbol *s = ops[1].vec_symbol(r);
        parse_assert(s && s->is_reg() &&
                     (s->type()->get_key().scalar_type() == B32_TYPE ||
                      s->type()->get_key().scalar_type() == S32_TYPE ||
                      s->type()->get_key().scalar_type() == U32_TYPE),
                     "TMA store coordinates must be 32-bit registers");
      }
    } else if (g_cluster_opcode.find("stmatrix.") == 0) {
      std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
      parse_assert(ops.size() == 2 && ops[0].is_memory_operand() &&
                   ops[1].is_vector() && ops[1].get_vect_nelem() == 4,
                   "stmatrix.x4 requires a shared address and four source registers");
      const symbol *address = ops[0].get_symbol();
      parse_assert(address && (address->is_shared() ||
                   (address->is_reg() &&
                    (address->type()->get_key().scalar_type() == B32_TYPE ||
                     address->type()->get_key().scalar_type() == U32_TYPE ||
                     address->type()->get_key().scalar_type() == B64_TYPE ||
                     address->type()->get_key().scalar_type() == U64_TYPE))),
                   "stmatrix address must be a shared symbol or 32/64-bit register");
      for (unsigned r = 0; r < 4; ++r) {
        const symbol *s = ops[1].vec_symbol(r);
        parse_assert(s && s->is_reg() &&
                     (s->type()->get_key().scalar_type() == B32_TYPE ||
                      s->type()->get_key().scalar_type() == U32_TYPE),
                     "stmatrix sources must be 32-bit registers");
      }
    } else {
      parse_assert(g_operands.empty() && !g_pred,
                   "fence.mbarrier_init requires no operands or predication");
    }
  }
  if (g_cluster_opcode.find("ld.shared::cluster.") == 0 ||
      g_cluster_opcode.find("st.shared::cluster.") == 0) {
    std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
    const bool load = g_opcode == LD_OP;
    parse_assert(ops.size() == 2 && ops[load ? 0 : 1].is_reg() &&
                 ops[load ? 1 : 0].is_memory_operand(),
                 "cluster shared access requires scalar register and memory operand");
  }
  if (g_opcode == CLUSTER_BARRIER_OP)
    parse_assert(g_operands.empty() && !g_pred,
                 "aligned cluster barrier requires all threads and no operands");
  PTX_PARSE_DPRINTF("add_instruction: %s",
                    ((g_opcode > 0) ? g_opcode_string[g_opcode] : "<label>"));
  assert(g_shader_core_config != 0);
  ptx_instruction *i = new ptx_instruction(
      g_opcode, g_pred, g_neg_pred, g_pred_mod, g_label, g_operands,
      g_return_var, g_options, g_wmma_options, g_scalar_type, g_space_spec,
      gpgpu_ctx->g_filename, ptx_get_lineno(scanner), linebuf,
      g_shader_core_config, gpgpu_ctx);
  i->ptx_cluster_opcode = g_cluster_opcode;
  if (!g_matrix_opcode.empty()) {
    i->ptx_matrix.opcode = g_matrix_opcode;
    if (g_opcode == LDMATRIX_OP) {
      i->ptx_matrix.kind = ptx_matrix_metadata::LDMATRIX;
      i->ptx_matrix.m = i->ptx_matrix.n = 8;
      i->ptx_matrix.matrices = g_operands.front().get_vect_nelem();
      i->ptx_matrix.transpose = g_matrix_opcode.find(".trans.") != std::string::npos;
      i->ptx_matrix.functional_supported = true;
    } else {
      i->ptx_matrix.kind = g_opcode == MMA_SYNC_OP
                              ? ptx_matrix_metadata::MMA_SYNC
                              : ptx_matrix_metadata::WGMMA;
      if (g_opcode == WGMMA_MMA_OP) {
        i->ptx_matrix.m = 64;
        i->ptx_matrix.n = g_matrix_opcode.find("n176") != std::string::npos ? 176 : 128;
        i->ptx_matrix.k = 16;
      } else if (g_opcode == MMA_SYNC_OP) {
        i->ptx_matrix.m = 16;
        i->ptx_matrix.n = 8;
        i->ptx_matrix.k = g_matrix_opcode.find("k8.") != std::string::npos ? 8 :
            g_matrix_opcode.find("k32.") != std::string::npos ? 32 : 16;
      }
    }
    if (i->ptx_matrix.kind == ptx_matrix_metadata::WGMMA) {
      auto &matrix = i->ptx_matrix;
      matrix.functional_supported = matrix.timing_supported = true;
      matrix.wgmma = g_opcode == WGMMA_MMA_OP ? ptx_matrix_metadata::MMA :
          g_opcode == WGMMA_FENCE_OP ? ptx_matrix_metadata::FENCE :
          g_opcode == WGMMA_COMMIT_OP ? ptx_matrix_metadata::COMMIT :
          ptx_matrix_metadata::WAIT;
      if (g_opcode == WGMMA_WAIT_OP)
        matrix.wait_group = g_operands.front().get_literal_value().u32;
      i->m_is_gmma_commit_group = g_opcode == WGMMA_COMMIT_OP;
    }
    if (!g_operands.empty() && g_operands.front().is_vector()) {
      const operand_info &dst = g_operands.front();
      for (unsigned r = 0; r < dst.get_vect_nelem(); ++r)
        i->ptx_matrix.accumulator_registers.push_back(dst.vec_symbol(r)->reg_num());
    }
  }
  if (!g_mbarrier_opcode.empty()) {
    i->ptx_mbarrier.opcode = g_mbarrier_opcode;
    i->ptx_mbarrier.functional_supported = true;
    i->ptx_mbarrier.timing_supported = false;
    if (g_mbarrier_opcode.find(".init.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::INIT;
    else if (g_mbarrier_opcode.find(".inval.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::INVALIDATE;
    else if (g_mbarrier_opcode.find(".arrive_drop.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::ARRIVE_DROP;
    else if (g_mbarrier_opcode.find(".arrive.expect_tx.") != std::string::npos ||
             g_mbarrier_opcode.find(".arrive.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::ARRIVE;
    else if (g_mbarrier_opcode.find(".expect_tx.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::EXPECT_TX;
    else if (g_mbarrier_opcode.find(".complete_tx.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::COMPLETE_TX;
    else if (g_mbarrier_opcode.find(".test_wait.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::TEST_WAIT;
    else if (g_mbarrier_opcode.find(".try_wait.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::TRY_WAIT;
    else if (g_mbarrier_opcode.find(".pending_count.") != std::string::npos)
      i->ptx_mbarrier.kind = ptx_mbarrier_op::PENDING_COUNT;
    i->ptx_mbarrier.parity = g_mbarrier_opcode.find(".parity.") != std::string::npos;
    switch (i->ptx_mbarrier.kind) {
      case ptx_mbarrier_op::INIT:
        i->set_syncs_op(SYNCS_INIT); break;
      case ptx_mbarrier_op::INVALIDATE:
        i->set_syncs_op(SYNCS_INVALIDATE); break;
      case ptx_mbarrier_op::EXPECT_TX:
        i->set_syncs_op(SYNCS_EXPECT_TX); break;
      case ptx_mbarrier_op::COMPLETE_TX:
        i->set_syncs_op(SYNCS_COMPELTE_TX); break;
      case ptx_mbarrier_op::ARRIVE:
        i->set_syncs_op(g_mbarrier_opcode.find(".expect_tx.") != std::string::npos
                            ? SYNCS_ARRIVE_EXPECT_TX : SYNCS_ARRIVE); break;
      case ptx_mbarrier_op::ARRIVE_DROP:
        i->set_syncs_op(SYNCS_ARRIVE_DROP); break;
      case ptx_mbarrier_op::TEST_WAIT:
        i->set_syncs_op(SYNCS_TEST_WAIT); break;
      case ptx_mbarrier_op::TRY_WAIT:
        i->set_syncs_op(SYNCS_TRY_WAIT); break;
      case ptx_mbarrier_op::PENDING_COUNT:
        i->set_syncs_op(SYNCS_PENDING_COUNT); break;
      default: break;
    }
  }
  if (!g_tma_opcode.empty()) {
    i->ptx_tma.opcode = g_tma_opcode;
    i->ptx_tma.functional_supported = true;
    if (g_tma_opcode.find(".prefetch.") != std::string::npos)
      i->ptx_tma.kind = ptx_tma_op::PREFETCH;
    else
      i->ptx_tma.kind = ptx_tma_op::LOAD;
    for (unsigned r = 1; r <= 5; ++r)
      if (g_tma_opcode.find("." + std::to_string(r) + "d") != std::string::npos)
        i->ptx_tma.rank = r;
    i->ptx_tma.multicast = g_tma_opcode.find(".multicast::cluster") != std::string::npos;
  }
  g_instructions.push_back(i);
  g_inst_lookup[gpgpu_ctx->g_filename][ptx_get_lineno(scanner)] = i;
  init_instruction_state();
}

void ptx_recognizer::add_variables() {
  PTX_PARSE_DPRINTF("add_variables");
  if (!g_operands.empty()) {
    assert(g_last_symbol != NULL);
    g_last_symbol->add_initializer(g_operands);
  }
  init_directive_state();
}

void ptx_recognizer::set_variable_type() {
  PTX_PARSE_DPRINTF("set_variable_type space_spec=%s scalar_type_spec=%s",
                    g_ptx_token_decode[g_space_spec.get_type()].c_str(),
                    g_ptx_token_decode[g_scalar_type_spec].c_str());
  parse_assert(g_space_spec != undefined_space,
               "variable has no space specification");
  parse_assert(
      g_scalar_type_spec != -1,
      "variable has no type information");  // need to extend for structs?
  g_var_type = g_current_symbol_table->add_type(
      g_space_spec, g_scalar_type_spec, g_vector_spec, g_alignment_spec,
      g_extern_spec);
}

bool ptx_recognizer::check_for_duplicates(const char *identifier) {
  const symbol *s = g_current_symbol_table->lookup(identifier);
  return (s != NULL);
}

// Returns padding that needs to be inserted ahead of address to make it aligned
// to min(size, maxalign)
/*
 * @param address the address in bytes
 * @param size the size of the memory to be allocated in bytes
 * @param maximum alignment in bytes. i.e. if size is too big then align to this
 * instead
 */
int pad_address(new_addr_type address, unsigned size, unsigned maxalign) {
  assert(size >= 0);
  assert(maxalign > 0);
  int alignto = maxalign;
  if (size < maxalign && (size & (size - 1)) == 0) {  // size is a power of 2
    alignto = size;
  }
  return alignto ? ((alignto - (address % alignto)) % alignto) : 0;
}

void ptx_recognizer::add_identifier(const char *identifier, int array_dim,
                                    unsigned array_ident) {
  if (g_func_decl && (g_func_info == NULL)) {
    // return variable decl...
    assert(g_add_identifier_cached__identifier == NULL);
    g_add_identifier_cached__identifier = strdup(identifier);
    g_add_identifier_cached__array_dim = array_dim;
    g_add_identifier_cached__array_ident = array_ident;
    return;
  }
  PTX_PARSE_DPRINTF("add_identifier \"%s\" (%u)", identifier, g_ident_add_uid);
  g_ident_add_uid++;
  type_info *type = g_var_type;
  type_info_key ti = type->get_key();
  int basic_type;
  int regnum;
  size_t num_bits;
  unsigned addr_pad;
  new_addr_type addr;
  ti.type_decode(num_bits, basic_type);
  if (g_func_decl && g_vector_spec != -1) {
    const unsigned lanes = g_vector_spec == V2_TYPE ? 2 :
                           g_vector_spec == V3_TYPE ? 3 : 4;
    num_bits *= lanes;
    g_size *= lanes;
  }

  bool duplicates = check_for_duplicates(identifier);
  if (duplicates) {
    symbol *s = g_current_symbol_table->lookup(identifier);
    g_last_symbol = s;
    if (g_func_decl) return;
    std::string msg = std::string(identifier) + " was declared previous at " +
                      s->decl_location() + " skipping new declaration";
    printf("GPGPU-Sim PTX: Warning %s\n", msg.c_str());
    return;
  }

  assert(g_var_type != NULL);
  switch (array_ident) {
    case ARRAY_IDENTIFIER: {
      const size_t element_bytes = (num_bits + 7) / 8;
      parse_assert(array_dim > 0 && element_bytes > 0 &&
                       size_t(array_dim) <=
                           std::numeric_limits<size_t>::max() / num_bits &&
                       size_t(array_dim) <=
                           std::numeric_limits<unsigned>::max() / element_bytes,
                   "array size exceeds unsigned symbol address range");
      type = g_current_symbol_table->get_array_type(type, array_dim);
      num_bits = array_dim * num_bits;
      break;
    }
    case ARRAY_IDENTIFIER_NO_DIM:
      type = g_current_symbol_table->get_array_type(type, (unsigned)-1);
      num_bits = 0;
      break;
    default:
      break;
  }
  g_last_symbol = g_current_symbol_table->add_variable(
      identifier, type, num_bits / 8, gpgpu_ctx->g_filename,
      ptx_get_lineno(scanner));
  switch (ti.get_memory_space().get_type()) {
    case reg_space: {
      regnum = g_current_symbol_table->next_reg_num();
      int arch_regnum = -1;
      for (int d = 0; d < strlen(identifier); d++) {
        if (isdigit(identifier[d])) {
          sscanf(identifier + d, "%d", &arch_regnum);
          break;
        }
      }
      if (strcmp(identifier, "%sp") == 0) {
        arch_regnum = 0;
      }
      g_last_symbol->set_regno(regnum, arch_regnum);
    } break;
    case shared_space:
      printf("GPGPU-Sim PTX: allocating shared region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_shared_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (shared memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_shared(num_bits / 8 + addr_pad);
      break;
    case sstarr_space:
      printf("GPGPU-Sim PTX: allocating sstarr region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_sstarr_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (sstarr memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_sstarr(num_bits / 8 + addr_pad);
      break;
    case const_space:
      if (array_ident == ARRAY_IDENTIFIER_NO_DIM) {
        printf(
            "GPGPU-Sim PTX: deferring allocation of constant region for \"%s\" "
            "(need size information)\n",
            identifier);
      } else {
        printf("GPGPU-Sim PTX: allocating constant region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_global_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx (global memory space) %u\n",
               addr + addr_pad, addr + addr_pad + num_bits / 8,
               g_const_alloc++);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);
      }
      if (g_current_symbol_table == g_global_symbol_table) {
        gpgpu_ctx->func_sim->g_constants.insert(identifier);
      }
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;
      break;
    case global_space:
      printf("GPGPU-Sim PTX: allocating global region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_global_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (global memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);
      gpgpu_ctx->func_sim->g_globals.insert(identifier);
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;
      break;
    case local_space:
      if (g_func_info == NULL) {
        printf("GPGPU-Sim PTX: allocating local region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx (local memory space)\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);
      } else {
        printf(
            "GPGPU-Sim PTX: allocating stack frame region for .local \"%s\" ",
            identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);
        g_func_info->set_framesize(g_current_symbol_table->get_local_next());
      }
      break;
    case tex_space:
      printf("GPGPU-Sim PTX: encountered texture directive %s.\n", identifier);
      break;
    case param_space_local:
      printf(
          "GPGPU-Sim PTX: allocating stack frame region for .param \"%s\" from "
          "0x%llx to 0x%llx\n",
          identifier, g_current_symbol_table->get_local_next(),
          g_current_symbol_table->get_local_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      g_last_symbol->set_address(g_current_symbol_table->get_local_next());
      g_current_symbol_table->alloc_local(num_bits / 8);
      g_func_info->set_framesize(g_current_symbol_table->get_local_next());
      break;
    case param_space_kernel:
      break;
    default:
      abort();
      break;
  }

  assert(!ti.is_param_unclassified());
  if (ti.is_param_kernel()) {
    bool is_ptr = (g_ptr_spec != undefined_space);
    g_func_info->add_param_name_type_size(g_entry_func_param_index, identifier,
                                          ti.scalar_type(), num_bits, is_ptr,
                                          g_ptr_spec);
    g_entry_func_param_index++;
  }
}

void ptx_recognizer::add_constptr(const char *identifier1,
                                  const char *identifier2, int offset) {
  symbol *s1 = g_current_symbol_table->lookup(identifier1);
  const symbol *s2 = g_current_symbol_table->lookup(identifier2);
  parse_assert(s1 != NULL, "'from' constant identifier does not exist.");
  parse_assert(s1 != NULL, "'to' constant identifier does not exist.");

  unsigned addr = s2->get_address();

  printf("GPGPU-Sim PTX: moving \"%s\" from 0x%llx to 0x%x (%s+%d)\n",
         identifier1, s1->get_address(), addr + offset, identifier2, offset);

  s1->set_address(addr + offset);
}

void ptx_recognizer::add_function_arg() {
  parse_assert(g_last_symbol != NULL && g_last_symbol->get_size_in_bytes() > 0,
               "function parameter must have a positive size");
  if (g_func_info) {
    PTX_PARSE_DPRINTF("add_function_arg \"%s\"", g_last_symbol->name().c_str());
    g_func_info->add_arg(g_last_symbol);
    unsigned alignment = (g_alignment_spec == -1) ? g_size : g_alignment_spec;
    parse_assert(function_info::valid_param_alignment(alignment),
                 "parameter alignment must be a power of two <= 1048576");
    try {
      g_func_info->add_config_param(g_last_symbol->get_size_in_bytes(), alignment);
    } catch (const std::exception &error) {
      parse_assert(false, "%s", error.what());
    }
  }
}

void ptx_recognizer::add_extern_spec() {
  PTX_PARSE_DPRINTF("add_extern_spec");
  g_extern_spec = 1;
}

void ptx_recognizer::add_alignment_spec(int spec) {
  PTX_PARSE_DPRINTF("add_alignment_spec");
  parse_assert(
      g_alignment_spec == -1,
      "multiple .align specifiers per variable declaration not allowed.");
  parse_assert(spec > 0 && function_info::valid_param_alignment(spec),
               ".align must be a positive power of two <= 1048576");
  g_alignment_spec = spec;
}

void ptx_recognizer::add_ptr_spec(enum _memory_space_t spec) {
  PTX_PARSE_DPRINTF("add_ptr_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  parse_assert(g_ptr_spec == undefined_space,
               "multiple ptr space specifiers not allowed.");
  parse_assert(
      spec == global_space or spec == local_space or spec == shared_space,
      "invalid space for ptr directive.");
  g_ptr_spec = spec;
}

void ptx_recognizer::add_space_spec(enum _memory_space_t spec, int value) {
  PTX_PARSE_DPRINTF("add_space_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  parse_assert(g_space_spec == undefined_space,
               "multiple space specifiers not allowed.");
  if (spec == param_space_unclassified) {
    if (g_func_decl) {
      if (g_entry_point == 1)
        g_space_spec = param_space_kernel;
      else
        g_space_spec = param_space_local;
    } else
      g_space_spec = param_space_unclassified;
  } else {
    g_space_spec = spec;
    if (g_space_spec == const_space) g_space_spec.set_bank((unsigned)value);
  }
}

void ptx_recognizer::add_vector_spec(int spec) {
  PTX_PARSE_DPRINTF("add_vector_spec");
  parse_assert(g_vector_spec == -1, "multiple vector specifiers not allowed.");
  g_vector_spec = spec;
}

void ptx_recognizer::add_scalar_type_spec(int type_spec) {
  if (type_spec == BF16_TYPE || type_spec == TF32_TYPE ||
      type_spec == E4M3_TYPE || type_spec == E5M2_TYPE)
    parse_assert(g_opcode != -1, "alternate floating formats are instruction types; declare bit registers");
  // save size of parameter
  switch (type_spec) {
    case B8_TYPE:
    case S8_TYPE:
    case U8_TYPE:
      g_size = 1;
      break;
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
    case F16_TYPE:
      g_size = 2;
      break;
    case BF16_TYPE:
      g_size = 2;
      break;
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
    case F32_TYPE:
      g_size = 4;
      break;
    case TF32_TYPE:
      g_size = 4;
      break;
    case E4M3_TYPE:
    case E5M2_TYPE:
      g_size = 1;
      break;
    case B64_TYPE:
    case BB64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      g_size = 8;
      break;
    case BB128_TYPE:
      g_size = 16;
      break;
  }
  PTX_PARSE_DPRINTF("add_scalar_type_spec \"%s\"",
                    g_ptx_token_decode[type_spec].c_str());
  g_scalar_type.push_back(type_spec);
  if (g_scalar_type.size() > 1) {
    parse_assert(
        (g_opcode == -1) || (g_opcode == CVT_OP) || (g_opcode == SET_OP) ||
            (g_opcode == SLCT_OP) || (g_opcode == TEX_OP) ||
            (g_opcode == MMA_OP) || (g_opcode == DP4A_OP) ||
            (g_opcode == VMIN_OP) || (g_opcode == VMAX_OP),
        "only cvt, set, slct, tex, vmin, vmax and dp4a can have more than one "
        "type specifier.");
  }
  g_scalar_type_spec = type_spec;
}

void ptx_recognizer::add_label(const char *identifier) {
  PTX_PARSE_DPRINTF("add_label");
  symbol *s = g_current_symbol_table->lookup(identifier);
  if (s != NULL) {
    g_label = s;
  } else {
    g_label = g_current_symbol_table->add_variable(
        identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
  }
}

void ptx_recognizer::add_opcode(int opcode) { g_opcode = opcode; }

void ptx_recognizer::begin_vector_operand(const char *identifier) {
  g_vector_operands.clear();
  g_vector_operands.push_back(identifier);
}

void ptx_recognizer::append_vector_operand(const char *identifier) {
  g_vector_operands.push_back(identifier);
}

void ptx_recognizer::finish_vector_operand() {
  parse_assert(!g_vector_operands.empty() && g_vector_operands.size() <= 128,
               "vector tuple must contain between 1 and 128 operands");
  if (g_matrix_opcode.empty() && g_tma_opcode.empty() &&
      g_cluster_opcode != "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group")
    parse_assert(g_vector_operands.size() == 1 || g_vector_operands.size() == 2 ||
                 g_vector_operands.size() == 3 || g_vector_operands.size() == 4 ||
                 g_vector_operands.size() == 8, "unsupported legacy vector tuple length");
  std::vector<const symbol *> symbols;
  for (std::vector<std::string>::const_iterator i = g_vector_operands.begin();
       i != g_vector_operands.end(); ++i) {
    const symbol *s = g_current_symbol_table->lookup(i->c_str());
    parse_assert(s != NULL, "vector component is not declared: %s", i->c_str());
    const symbol *null_op = g_current_symbol_table->lookup("_");
    symbols.push_back(s == null_op ? NULL : s);
  }
  g_operands.push_back(operand_info(symbols, gpgpu_ctx));
  g_vector_operands.clear();
}

void ptx_recognizer::add_matrix_opcode(const char *opcode) {
  std::string text(opcode);
  if (std::regex_match(text, std::regex(
          "ldmatrix\\.sync\\.aligned\\.m8n8\\.x[124](\\.trans)?\\.shared(::cta)?\\.b16"))) {
    g_opcode = LDMATRIX_OP;
  } else if (text == "wgmma.fence.sync.aligned") {
    g_opcode = WGMMA_FENCE_OP;
  } else if (text == "wgmma.commit_group.sync.aligned") {
    g_opcode = WGMMA_COMMIT_OP;
  } else if (text == "wgmma.wait_group.sync.aligned") {
    g_opcode = WGMMA_WAIT_OP;
  } else if (std::regex_match(text, std::regex(
                 "wgmma\\.mma_async\\.sync\\.aligned\\.m64n(128|176)k16\\.f32\\.f16\\.f16"))) {
    g_opcode = WGMMA_MMA_OP;
  } else if (text == "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" ||
             text == "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" ||
             text == "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" ||
             text == "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" ||
             text == "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32") {
    g_opcode = MMA_SYNC_OP;
  } else {
    parse_assert(0, "unsupported matrix opcode/modifier: %s", opcode);
  }
  g_options.clear();
  g_wmma_options.clear();
  g_scalar_type.clear();
  g_matrix_opcode = text;
  if (g_opcode == LDMATRIX_OP) {
    g_space_spec = shared_space;
    g_scalar_type.push_back(B16_TYPE);
  }
}

void ptx_recognizer::add_mbarrier_opcode(const char *opcode) {
  std::string text(opcode);
  const auto has = [&](const char *part) {
    return text.find(part) != std::string::npos;
  };
  ptx_mbarrier_op kind = ptx_mbarrier_op::NONE;
  if (has(".init.")) kind = ptx_mbarrier_op::INIT;
  else if (has(".inval.")) kind = ptx_mbarrier_op::INVALIDATE;
  else if (has(".arrive_drop.")) kind = ptx_mbarrier_op::ARRIVE_DROP;
  else if (has(".arrive.expect_tx.") || has(".arrive.")) kind = ptx_mbarrier_op::ARRIVE;
  else if (has(".expect_tx.")) kind = ptx_mbarrier_op::EXPECT_TX;
  else if (has(".complete_tx.")) kind = ptx_mbarrier_op::COMPLETE_TX;
  else if (has(".test_wait.")) kind = ptx_mbarrier_op::TEST_WAIT;
  else if (has(".try_wait.")) kind = ptx_mbarrier_op::TRY_WAIT;
  else if (has(".pending_count.")) kind = ptx_mbarrier_op::PENDING_COUNT;
  parse_assert(kind != ptx_mbarrier_op::NONE,
               "unsupported mbarrier opcode/modifier: %s", opcode);
  parse_assert(text.find(".b64") != std::string::npos,
               "mbarrier requires .b64: %s", opcode);
  parse_assert(text.find(".shared") != std::string::npos,
               "mbarrier requires shared address space: %s", opcode);
  if (has("::cluster"))
    parse_assert(kind == ptx_mbarrier_op::ARRIVE ||
                     kind == ptx_mbarrier_op::ARRIVE_DROP ||
                     kind == ptx_mbarrier_op::EXPECT_TX ||
                     kind == ptx_mbarrier_op::COMPLETE_TX,
                 "cluster mbarrier wait is unsupported: %s", opcode);
  g_opcode = MBARRIER_OP;
  g_mbarrier_opcode = text;
  g_matrix_opcode.clear();
  g_options.clear();
  g_wmma_options.clear();
  g_scalar_type.clear();
  g_scalar_type.push_back(B64_TYPE);
  g_space_spec = shared_space;
}

void ptx_recognizer::add_tma_opcode(const char *opcode) {
  const std::string text(opcode);
  if (text == "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group" ||
      text == "cp.async.bulk.commit_group" || text == "cp.async.bulk.wait_group.read") {
    // Store completion uses bulk groups, not load mbarrier metadata.
    g_opcode = HOPPER_FRONTEND_ONLY_OP;
    g_cluster_opcode = text;
    g_scalar_type.clear();
    g_scalar_type.push_back(B64_TYPE);
    return;
  }
  const bool load = std::regex_match(text, std::regex(
      "cp\\.async\\.bulk\\.tensor\\.[1-5]d\\.shared::cluster\\.global(\\.tile)?\\.mbarrier::complete_tx::bytes(\\.multicast::cluster)?(\\.L2::cache_hint)?"));
  const bool prefetch = std::regex_match(text, std::regex(
      "cp\\.async\\.bulk\\.prefetch\\.tensor\\.[1-5]d\\.L2\\.global"));
  parse_assert(load || prefetch, "unsupported TMA opcode/modifier: %s", opcode);
  g_opcode = TMA_PTX_OP;
  g_tma_opcode = text;
  g_scalar_type.clear();
  g_scalar_type.push_back(B64_TYPE);
  g_space_spec = global_space;
}

void ptx_recognizer::add_cluster_opcode(const char *opcode) {
  const std::string text(opcode);
  g_scalar_type.clear();
  if (text == "elect.sync") {
    // Frontend readiness only: execution must fail until warp rendezvous exists.
    g_opcode = ELECT_OP;
    g_scalar_type.push_back(U32_TYPE);
  } else if (text == "prefetch.tensormap" || text == "prefetch.const.tensormap" ||
             text == "prefetch.param.tensormap") {
    g_opcode = PREFETCH_TENSORMAP_OP;
    g_scalar_type.push_back(B64_TYPE);
  } else if (text == "fence.mbarrier_init.release.cluster") {
    g_opcode = HOPPER_FRONTEND_ONLY_OP;
    g_scalar_type.push_back(B64_TYPE);
  } else if (text == "stmatrix.sync.aligned.x4.m8n8.shared.b16") {
    g_opcode = HOPPER_FRONTEND_ONLY_OP;
    g_scalar_type.push_back(B16_TYPE);
  } else if (text == "griddepcontrol.wait") {
    g_opcode = HOPPER_FRONTEND_ONLY_OP;
    g_scalar_type.push_back(B64_TYPE);
  } else if (text == "mapa.shared::cluster.u32" || text == "mapa.shared::cluster.u64") {
    g_opcode = MAPA_OP;
    g_scalar_type.push_back(text.find(".u64") != std::string::npos ? U64_TYPE : U32_TYPE);
  } else if (text == "getctarank.shared::cluster.u32" || text == "getctarank.shared::cluster.u64") {
    g_opcode = GETCTARANK_OP;
    g_scalar_type.push_back(text.find(".u64") != std::string::npos ? U64_TYPE : U32_TYPE);
  } else if (text == "setmaxnreg.inc.sync.aligned.u32" ||
             text == "setmaxnreg.dec.sync.aligned.u32") {
    g_opcode = SETMAXNREG_OP;
    g_scalar_type.push_back(U32_TYPE);
  } else if (std::regex_match(text, std::regex(
                 "barrier\\.cluster\\.(arrive(\\.(release|relaxed))?|wait(\\.acquire)?)\\.aligned"))) {
    g_opcode = CLUSTER_BARRIER_OP;
  } else if (std::regex_match(text, std::regex("(ld|st)\\.shared::cluster\\.(u32|b32|u64|b64)"))) {
    g_opcode = text.find("ld.") == 0 ? LD_OP : ST_OP;
    g_scalar_type.push_back(text.find("64") != std::string::npos ? U64_TYPE : U32_TYPE);
    g_space_spec = shared_space;
  } else {
    const bool readiness_opcode = text.find("elect.") == 0 ||
        text.find("prefetch.") == 0 || text.find("fence.mbarrier_init.") == 0 ||
        text.find("stmatrix.") == 0 || text.find("griddepcontrol.") == 0;
    parse_assert(0, readiness_opcode ? "unsupported Hopper opcode/modifier: %s"
                                    : "unsupported cluster opcode/modifier: %s", opcode);
  }
  g_cluster_opcode = text;
  g_options.clear();
  g_wmma_options.clear();
}

void ptx_recognizer::validate_tma_operands() {
  if (g_tma_opcode.empty()) return;
  std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
  const bool prefetch = g_tma_opcode.find(".prefetch.") != std::string::npos;
  const bool multicast = g_tma_opcode.find(".multicast::cluster") != std::string::npos;
  const bool cache_hint = g_tma_opcode.find(".L2::cache_hint") != std::string::npos;
  const unsigned coord_pos = prefetch ? 1 : 2;
  parse_assert(ops.size() == (prefetch ? 2 : 4 + unsigned(multicast) + unsigned(cache_hint)),
               "TMA tensor operand count mismatch");
  parse_assert(ops[0].is_memory_operand() || ops[0].is_reg(),
               "TMA requires an address operand");
  if (!prefetch)
    parse_assert(ops[1].is_reg() || ops[1].is_memory_operand(),
                 "TMA descriptor requires an address operand");
  parse_assert(ops[coord_pos].is_vector(), "TMA coordinates require a vector tuple");
  const unsigned rank = g_tma_opcode[g_tma_opcode.find("tensor.") + 7] - '0';
  parse_assert(ops[coord_pos].get_vect_nelem() == rank, "TMA coordinate rank mismatch");
  for (unsigned r = 0; r < rank; ++r) {
    const symbol *s = ops[coord_pos].vec_symbol(r);
    parse_assert(s->is_reg() && (s->type()->get_key().scalar_type() == B32_TYPE ||
                 s->type()->get_key().scalar_type() == S32_TYPE ||
                 s->type()->get_key().scalar_type() == U32_TYPE),
                 "TMA coordinates require 32-bit registers");
  }
  if (!prefetch) {
    parse_assert(ops[3].is_memory_operand(), "TMA completion requires an mbarrier address");
    if (multicast) parse_assert(ops[4].is_reg() || ops[4].is_literal(),
                               "TMA multicast mask requires a scalar");
    if (cache_hint) parse_assert(ops.back().is_reg() || ops.back().is_literal(),
                                "TMA cache hint requires a scalar");
  }
}

void ptx_recognizer::validate_matrix_operands() {
  if (g_matrix_opcode.empty()) return;
  std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
  if (g_matrix_opcode.find("wgmma.") == 0)
    parse_assert(!g_pred, "predicated wgmma is unsupported: full warpgroup required");
  const auto tuple = [&](unsigned pos, unsigned count, int type) {
    parse_assert(pos < ops.size() && ops[pos].is_vector() &&
                 ops[pos].get_vect_nelem() == count, "matrix register tuple length mismatch");
    std::set<const symbol *> seen;
    for (unsigned r = 0; r < count; ++r) {
      const symbol *s = ops[pos].vec_symbol(r);
      parse_assert(s && s->is_reg() && s->type()->get_key().scalar_type() == type,
                   "matrix tuple requires registers of the specified bit width/type");
      parse_assert(seen.insert(s).second, "duplicate register in matrix tuple");
    }
  };
  const auto reg = [&](unsigned pos, unsigned bits) {
    parse_assert(pos < ops.size() && ops[pos].is_reg(), "matrix operand requires a register");
    size_t width; int basic;
    ops[pos].get_symbol()->type()->get_key().type_decode(width, basic);
    parse_assert(width == bits, "matrix register width mismatch");
  };
  if (g_opcode == LDMATRIX_OP) {
    parse_assert(ops.size() == 2 && ops[1].is_memory_operand(), "ldmatrix requires a tuple and shared address");
    unsigned count = g_matrix_opcode[g_matrix_opcode.find(".x") + 2] - '0';
    tuple(0, count, B32_TYPE);
    const symbol *address = ops[1].get_symbol();
    parse_assert(address && (address->is_reg() || address->is_shared()), "ldmatrix address must be a register or shared symbol");
    if (address->is_reg()) {
      const int type = address->type()->get_key().scalar_type();
      parse_assert(type == B32_TYPE || type == B64_TYPE ||
                       type == U32_TYPE || type == U64_TYPE ||
                       type == S32_TYPE || type == S64_TYPE,
                   "ldmatrix address register must be a 32/64-bit integer or bit register");
    }
    parse_assert(!g_pred, "predicated ldmatrix is not supported");
  } else if (g_opcode == MMA_SYNC_OP) {
    parse_assert(ops.size() == 4, "mma.sync requires four tuples");
    tuple(0, 4, F32_TYPE);
    bool tf32 = g_matrix_opcode.find("tf32") != std::string::npos;
    tuple(1, tf32 ? 4 : 4, B32_TYPE);
    tuple(2, 2, B32_TYPE);
    tuple(3, 4, F32_TYPE);
  } else if (g_opcode == WGMMA_MMA_OP) {
    bool registers = ops.size() > 1 && ops[1].is_vector();
    parse_assert(ops.size() == (registers ? 7 : 8), "wgmma operand count mismatch");
    tuple(0, g_matrix_opcode.find("n176") != std::string::npos ? 88 : 64, F32_TYPE);
    if (registers) tuple(1, 4, B32_TYPE);
    else reg(1, 64);
    reg(2, 64);
    parse_assert(ops[3].get_type() == int_t || (ops[3].is_reg() &&
        ops[3].get_symbol()->type()->get_key().scalar_type() == PRED_TYPE),
        "wgmma scale-d requires a predicate or 0/1 immediate");
    if (ops[3].get_type() == int_t)
      parse_assert(ops[3].get_literal_value().s64 == 0 ||
          ops[3].get_literal_value().s64 == 1, "wgmma scale-d must be 0 or 1");
    for (unsigned p = 4; p < ops.size(); ++p) {
      parse_assert(ops[p].get_type() == int_t, "wgmma scale/transpose requires an immediate");
      long long value = ops[p].get_literal_value().s64;
      parse_assert(p < 6 ? (value == -1 || value == 1) : (value == 0 || value == 1),
                   "wgmma scale/transpose immediate out of range");
    }
  } else if (g_opcode == WGMMA_WAIT_OP) {
    parse_assert(ops.size() == 1 && ops[0].get_type() == int_t, "wgmma.wait_group requires one immediate");
    long long count = ops[0].get_literal_value().s64;
    parse_assert(count >= 0 && count <= 7, "wgmma.wait_group immediate must be 0..7");
  } else {
    parse_assert(ops.empty(), "wgmma fence/commit take no operands");
  }
}

void ptx_recognizer::validate_mbarrier_operands() {
  if (g_mbarrier_opcode.empty()) return;
  std::vector<operand_info> ops(g_operands.begin(), g_operands.end());
  const auto address = [&](unsigned pos) {
    parse_assert(pos < ops.size() && ops[pos].is_memory_operand(),
                 "mbarrier requires a shared address operand");
    parse_assert(ops[pos].get_symbol() && ops[pos].get_symbol()->type(),
                 "mbarrier address has no symbol");
  };
  const auto value = [&](unsigned pos) {
    parse_assert(pos < ops.size() &&
                     (ops[pos].is_reg() || ops[pos].is_literal()),
                 "mbarrier count requires a register or immediate");
  };
  const bool returns_state = g_mbarrier_opcode.find(".cluster") == std::string::npos;
  const bool parity = g_mbarrier_opcode.find(".parity.") != std::string::npos;
  if (g_mbarrier_opcode.find(".init.") != std::string::npos) {
    parse_assert(ops.size() == 2, "mbarrier.init requires address and count");
    address(0); value(1);
  } else if (g_mbarrier_opcode.find(".inval.") != std::string::npos) {
    parse_assert(ops.size() == 1, "mbarrier.inval requires one address"); address(0);
  } else if (g_mbarrier_opcode.find(".arrive_drop.") != std::string::npos ||
             g_mbarrier_opcode.find(".arrive.") != std::string::npos) {
    const bool expect = g_mbarrier_opcode.find(".arrive.expect_tx.") != std::string::npos;
    parse_assert(ops.size() == (expect ? 3 : 2),
                 "mbarrier arrive operand count mismatch");
    if (returns_state)
      parse_assert(ops[0].is_reg() || ops[0].is_non_arch_reg(),
                   "mbarrier arrive state must be a .b64 register or sink");
    address(returns_state ? 1 : 0);
    if (expect) value(2);
  } else if (g_mbarrier_opcode.find(".expect_tx.") != std::string::npos ||
             g_mbarrier_opcode.find(".complete_tx.") != std::string::npos) {
    parse_assert(ops.size() == 2, "mbarrier transaction operation requires address and count");
    address(0); value(1);
  } else if (g_mbarrier_opcode.find(".test_wait.") != std::string::npos ||
             g_mbarrier_opcode.find(".try_wait.") != std::string::npos) {
    parse_assert(ops.size() == 3 || ops.size() == 4,
                 "mbarrier wait requires predicate, address, and state");
    parse_assert(ops[0].is_reg(), "mbarrier wait result must be a predicate register");
    address(1); value(2);
    if (parity) parse_assert(ops[2].is_literal() || ops[2].is_reg(),
                             "mbarrier parity requires a 32-bit value");
    if (ops.size() == 4) value(3);
  } else if (g_mbarrier_opcode.find(".pending_count.") != std::string::npos) {
    parse_assert(ops.size() == 2, "mbarrier.pending_count requires result and address");
    parse_assert(ops[0].is_reg(), "mbarrier.pending_count result must be a register");
    address(1);
  }
}

void ptx_recognizer::add_pred(const char *identifier, int neg,
                              int predModifier) {
  PTX_PARSE_DPRINTF("add_pred");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    std::string msg =
        std::string("predicate \"") + identifier + "\" has no declaration.";
    parse_error(msg.c_str());
  }
  g_pred = s;
  g_neg_pred = neg;
  g_pred_mod = predModifier;
}

void ptx_recognizer::add_option(int option) {
  PTX_PARSE_DPRINTF("add_option");
  if (g_func_decl && g_opcode == -1 &&
      (option == V2_TYPE || option == V3_TYPE || option == V4_TYPE)) {
    add_vector_spec(option);
    return;
  }
  g_options.push_back(option);
}
void ptx_recognizer::add_wmma_option(int option) {
  PTX_PARSE_DPRINTF("add_option");
  g_wmma_options.push_back(option);
}
void ptx_recognizer::add_double_operand(const char *d1, const char *d2) {
  // operands that access two variables.
  // eg. s[$ofs1+$r0], g[$ofs1+=$r0]
  // TODO: Not sure if I'm going to use this for storing to two destinations or
  // not.

  PTX_PARSE_DPRINTF("add_double_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  parse_assert(s1 != NULL && s2 != NULL, "component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, gpgpu_ctx));
}

void ptx_recognizer::add_1vector_operand(const char *d1) {
  // handles the single element vector operand ({%v1}) found in tex.1d
  // instructions
  PTX_PARSE_DPRINTF("add_1vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  parse_assert(s1 != NULL, "component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, NULL, NULL, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_2vector_operand(const char *d1, const char *d2) {
  PTX_PARSE_DPRINTF("add_2vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  parse_assert(s1 != NULL && s2 != NULL,
               "v2 component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, NULL, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_3vector_operand(const char *d1, const char *d2,
                                         const char *d3) {
  PTX_PARSE_DPRINTF("add_3vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL,
               "v3 component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, s3, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_4vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4) {
  PTX_PARSE_DPRINTF("add_4vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  const symbol *s4 = g_current_symbol_table->lookup(d4);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL,
               "v4 component(s) missing declarations.");
  const symbol *null_op = g_current_symbol_table->lookup("_");
  if (s2 == null_op) s2 = NULL;
  if (s3 == null_op) s3 = NULL;
  if (s4 == null_op) s4 = NULL;
  g_operands.push_back(operand_info(s1, s2, s3, s4, gpgpu_ctx));
}
void ptx_recognizer::add_8vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4,
                                         const char *d5, const char *d6,
                                         const char *d7, const char *d8) {
  PTX_PARSE_DPRINTF("add_8vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  const symbol *s4 = g_current_symbol_table->lookup(d4);
  const symbol *s5 = g_current_symbol_table->lookup(d5);
  const symbol *s6 = g_current_symbol_table->lookup(d6);
  const symbol *s7 = g_current_symbol_table->lookup(d7);
  const symbol *s8 = g_current_symbol_table->lookup(d8);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL &&
                   s5 != NULL && s6 != NULL && s7 != NULL && s8 != NULL,
               "v4 component(s) missing declarations.");
  const symbol *null_op = g_current_symbol_table->lookup("_");
  if (s2 == null_op) s2 = NULL;
  if (s3 == null_op) s3 = NULL;
  if (s4 == null_op) s4 = NULL;
  if (s5 == null_op) s5 = NULL;
  if (s6 == null_op) s6 = NULL;
  if (s7 == null_op) s7 = NULL;
  if (s8 == null_op) s8 = NULL;
  g_operands.push_back(operand_info(s1, s2, s3, s4, s5, s6, s7, s8, gpgpu_ctx));
}

void ptx_recognizer::add_builtin_operand(int builtin, int dim_modifier) {
  PTX_PARSE_DPRINTF("add_builtin_operand");
  if (builtin >= CLUSTERID_REG && builtin <= IS_EXPLICIT_CLUSTER_REG) {
    const bool scalar = builtin >= CLUSTER_CTARANK_REG;
    parse_assert(scalar ? dim_modifier == -1 : dim_modifier >= 0,
                 "cluster special register dimension modifier mismatch");
  }
  g_operands.push_back(operand_info(builtin, dim_modifier, gpgpu_ctx));
}

void ptx_recognizer::add_memory_operand() {
  PTX_PARSE_DPRINTF("add_memory_operand");
  assert(!g_operands.empty());
  g_operands.back().make_memory_operand();
}

/*TODO: add other memory locations*/
void ptx_recognizer::change_memory_addr_space(const char *identifier) {
  /*0 = N/A, not reading from memory
   *1 = global memory
   *2 = shared memory
   *3 = const memory segment
   *4 = local memory segment
   */

  bool recognizedType = false;

  PTX_PARSE_DPRINTF("change_memory_addr_space");
  assert(!g_operands.empty());
  if (!strcmp(identifier, "g")) {
    g_operands.back().set_addr_space(global_space);
    recognizedType = true;
  }
  if (!strcmp(identifier, "s")) {
    g_operands.back().set_addr_space(shared_space);
    recognizedType = true;
  }
  // For constants, check if the first character is 'c'
  char c[2];
  strncpy(c, identifier, 1);
  c[1] = '\0';
  if (!strcmp(c, "c")) {
    g_operands.back().set_addr_space(const_space);
    parse_assert(g_current_symbol_table->lookup(identifier) != NULL,
                 "Constant was not defined.");
    g_operands.back().set_const_mem_offset(
        g_current_symbol_table->lookup(identifier)->get_address());
    recognizedType = true;
  }
  // For local memory, check if the first character is 'l'
  char l[2];
  strncpy(l, identifier, 1);
  l[1] = '\0';
  if (!strcmp(l, "l")) {
    g_operands.back().set_addr_space(local_space);
    // parse_assert(g_current_symbol_table->lookup(identifier) != NULL, "Local
    // memory segment was not defined.");
    // g_operands.back().set_const_mem_offset(g_current_symbol_table->lookup(identifier)->get_address());
    recognizedType = true;
  }

  parse_assert(recognizedType, "Error: unrecognized memory type.");
}

void ptx_recognizer::change_operand_lohi(int lohi) {
  /*0 = N/A, read entire operand
   *1 = lo, reading from lowest bits
   *2 = hi, reading from highest bits
   */

  PTX_PARSE_DPRINTF("change_operand_lohi");
  assert(!g_operands.empty());

  g_operands.back().set_operand_lohi(lohi);
}

void ptx_recognizer::set_immediate_operand_type() {
  PTX_PARSE_DPRINTF("set_immediate_operand_type");
  assert(!g_operands.empty());
  g_operands.back().set_immediate_addr();
}

void ptx_recognizer::change_double_operand_type(int operand_type) {
  /*
   *-3 = reg / reg (set instruction, but both get same value)
   *-2 = reg | reg (cvt instruction)
   *-1 = reg | reg (set instruction)
   *0 = N/A, default
   *1 = reg + reg
   *2 = reg += reg
   *3 = reg += immediate
   */

  PTX_PARSE_DPRINTF("change_double_operand_type");
  assert(!g_operands.empty());

  // For double destination operands, ensure valid instruction
  if (operand_type == -1 || operand_type == -2) {
    if ((g_opcode == SET_OP) || (g_opcode == SETP_OP))
      g_operands.back().set_double_operand_type(-1);
    else
      g_operands.back().set_double_operand_type(-2);
  } else if (operand_type == -3) {
    if (g_opcode == SET_OP || g_opcode == MAD_OP)
      g_operands.back().set_double_operand_type(operand_type);
    else
      parse_assert(0, "Error: Unsupported use of double destination operand.");
  } else {
    g_operands.back().set_double_operand_type(operand_type);
  }
}

void ptx_recognizer::change_operand_neg() {
  PTX_PARSE_DPRINTF("change_operand_neg");
  assert(!g_operands.empty());

  g_operands.back().set_operand_neg();
}

void ptx_recognizer::add_literal_int(int value) {
  PTX_PARSE_DPRINTF("add_literal_int");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_literal_int64(unsigned long long value) {
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_literal_float(float value) {
  PTX_PARSE_DPRINTF("add_literal_float");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_literal_double(double value) {
  PTX_PARSE_DPRINTF("add_literal_double");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_scalar_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_scalar_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    if (g_opcode == BRA_OP || g_opcode == CALLP_OP) {
      // forward branch target...
      s = g_current_symbol_table->add_variable(
          identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
    } else {
      std::string msg =
          std::string("operand \"") + identifier + "\" has no declaration.";
      parse_error(msg.c_str());
    }
  }
  g_operands.push_back(operand_info(s, gpgpu_ctx));
}

void ptx_recognizer::add_neg_pred_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_neg_pred_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    s = g_current_symbol_table->add_variable(
        identifier, NULL, 1, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
  }
  operand_info op(s, gpgpu_ctx);
  op.set_neg_pred();
  g_operands.push_back(op);
}

void ptx_recognizer::add_address_operand(const char *identifier, int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    std::string msg =
        std::string("operand \"") + identifier + "\" has no declaration.";
    parse_error(msg.c_str());
  }
  g_operands.push_back(operand_info(s, offset, gpgpu_ctx));
}

void ptx_recognizer::add_address_operand2(int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");
  g_operands.push_back(operand_info((unsigned)offset, gpgpu_ctx));
}

void ptx_recognizer::add_array_initializer() {
  g_last_symbol->add_initializer(g_operands);
}

void ptx_recognizer::add_version_info(float ver, unsigned ext) {
  g_global_symbol_table->set_ptx_version(ver, ext);
}

void ptx_recognizer::add_file(unsigned num, const char *filename) {
  if (gpgpu_ctx->g_filename == NULL) {
    char *b = strdup(filename);
    char *l = b;
    char *n = b;
    while (*n != '\0') {
      if (*n == '/') l = n + 1;
      n++;
    }

    char *p = strtok(l, ".");
    char buf[1024];
    snprintf(buf, 1024, "%s.ptx", p);

    char *q = strtok(NULL, ".");
    if (q && !strcmp(q, "cu")) {
      gpgpu_ctx->g_filename = strdup(buf);
    }

    free(b);
  }

  g_current_symbol_table = g_global_symbol_table;
}

void *ptx_recognizer::reset_symtab() {
  void *result = g_current_symbol_table;
  g_current_symbol_table = g_global_symbol_table;
  return result;
}

void ptx_recognizer::set_symtab(void *symtab) {
  g_current_symbol_table = (symbol_table *)symtab;
}

void ptx_recognizer::add_pragma(const char *str) {
  printf("GPGPU-Sim PTX: Warning -- ignoring pragma '%s'\n", str);
}

void ptx_recognizer::version_header(double a) {}  // intentional dummy function

void ptx_recognizer::target_header(char *a) {
  g_global_symbol_table->set_sm_target(a, NULL, NULL);
}

void ptx_recognizer::target_header2(char *a, char *b) {
  g_global_symbol_table->set_sm_target(a, b, NULL);
}

void ptx_recognizer::target_header3(char *a, char *b, char *c) {
  g_global_symbol_table->set_sm_target(a, b, c);
}

void ptx_recognizer::maxnt_id(int x, int y, int z) {
  g_func_info->set_maxnt_id(x * y * z);
}

void ptx_recognizer::func_header(const char *a) {}  // intentional dummy
                                                    // function
void ptx_recognizer::func_header_info(const char *a) {
}  // intentional dummy function
void ptx_recognizer::func_header_info_int(const char *a, int b) {
}  // intentional dummy function
