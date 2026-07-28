/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include "iree_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/runtime/api.h"

/* ------------------------------------------------------------------ errors */

static char g_error[512];

static void set_error_status(const char *what, iree_status_t status) {
  char buffer[400];
  iree_host_size_t length = 0;
  if (!iree_status_format(status, sizeof(buffer), buffer, &length)) {
    buffer[0] = '\0';
  }
  snprintf(g_error, sizeof(g_error), "%s: %s", what, buffer);
  iree_status_ignore(status);
}

static void set_error_msg(const char *msg) {
  snprintf(g_error, sizeof(g_error), "%s", msg);
}

const char *xtc_iree_last_error(void) {
  return g_error[0] ? g_error : NULL;
}

/* ------------------------------------------------------------------ context */

typedef struct {
  void *host_ptr;         /* where to copy this output after each invoke */
  iree_device_size_t byte_length;
} xtc_output_slot_t;

typedef struct {
  iree_runtime_instance_t *instance;
  iree_hal_device_t *device;
  iree_runtime_session_t *session;
  iree_runtime_call_t call;

  iree_hal_buffer_view_t **input_views; /* [n_inputs], created once at setup */
  int n_inputs;

  xtc_output_slot_t *outputs; /* [n_outputs] */
  int n_outputs;
} xtc_iree_ctx_t;

/* Parse a descriptor's element type and compute its dense byte length. */
static iree_status_t desc_byte_length(const xtc_ndarray_desc_t *desc,
                                      iree_hal_element_type_t *out_element_type,
                                      iree_device_size_t *out_byte_length) {
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_parse_element_type(
      iree_make_cstring_view(desc->dtype), &element_type));
  iree_device_size_t count = 1;
  for (int i = 0; i < desc->rank; ++i) {
    count *= (iree_device_size_t)desc->shape[i];
  }
  *out_element_type = element_type;
  *out_byte_length = count * iree_hal_element_dense_byte_count(element_type);
  return iree_ok_status();
}

/* Build one host-resident input as an IREE buffer view (allocate + copy). The
 * copy happens at setup, outside any timed region. */
static iree_status_t make_input_view(iree_hal_device_t *device,
                                     iree_hal_allocator_t *allocator,
                                     const xtc_ndarray_desc_t *desc,
                                     iree_hal_buffer_view_t **out_view) {
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(desc_byte_length(desc, &element_type, &byte_length));

  iree_hal_dim_t *shape =
      (iree_hal_dim_t *)malloc((size_t)desc->rank * sizeof(iree_hal_dim_t));
  if (!shape) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  for (int i = 0; i < desc->rank; ++i) {
    shape[i] = (iree_hal_dim_t)desc->shape[i];
  }

  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, (iree_host_size_t)desc->rank, shape, element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(desc->data, (iree_host_size_t)byte_length),
      out_view);
  free(shape);
  return status;
}

void *xtc_iree_setup(const char *vmfb_path, const char *entry_function,
                     int single_thread, const xtc_ndarray_desc_t *inputs,
                     int n_inputs, const xtc_ndarray_desc_t *outputs,
                     int n_outputs) {
  g_error[0] = '\0';
  iree_allocator_t host_allocator = iree_allocator_system();

  xtc_iree_ctx_t *ctx = (xtc_iree_ctx_t *)calloc(1, sizeof(*ctx));
  if (!ctx) {
    set_error_msg("out of memory allocating context");
    return NULL;
  }

  /* Create the runtime instance with all available HAL drivers registered. */
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, host_allocator, &ctx->instance);

  /* Pick the CPU device: local-sync runs inline, local-task uses a thread pool. */
  if (iree_status_is_ok(status)) {
    const char *driver = single_thread ? "local-sync" : "local-task";
    status = iree_runtime_instance_try_create_default_device(
        ctx->instance, iree_make_cstring_view(driver), &ctx->device);
  }

  /* Open a session on that device to hold loaded modules and state. */
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        ctx->instance, &session_options, ctx->device,
        iree_runtime_instance_host_allocator(ctx->instance), &ctx->session);
  }

  /* Load the compiled bytecode module (.vmfb) into the session. */
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(ctx->session,
                                                                   vmfb_path);
  }

  /* Resolve the entry function and prepare a reusable call handle. */
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        ctx->session, iree_make_cstring_view(entry_function), &ctx->call);
  }

  /* Build the input buffer views once. */
  if (iree_status_is_ok(status)) {
    /* Allocate the array of view pointers (min 1 to avoid a zero-size calloc). */
    ctx->n_inputs = n_inputs;
    ctx->input_views = (iree_hal_buffer_view_t **)calloc(
        (size_t)(n_inputs > 0 ? n_inputs : 1), sizeof(*ctx->input_views));
    /* Grab the session's device allocator to back the input buffers. */
    iree_hal_allocator_t *device_allocator =
        iree_runtime_session_device_allocator(ctx->session);
    /* Allocate each device buffer and copy its host data in (untimed). */
    for (int i = 0; i < n_inputs && iree_status_is_ok(status); ++i) {
      status = make_input_view(ctx->device, device_allocator, &inputs[i],
                               &ctx->input_views[i]);
    }
  }

  /* Record where outputs must be copied back after each invocation. */
  if (iree_status_is_ok(status)) {
    /* Allocate the slot array (min 1 to avoid a zero-size calloc). */
    ctx->n_outputs = n_outputs;
    ctx->outputs = (xtc_output_slot_t *)calloc(
        (size_t)(n_outputs > 0 ? n_outputs : 1), sizeof(*ctx->outputs));
    for (int i = 0; i < n_outputs; ++i) {
      /* Compute the dense byte length expected for this output. */
      iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
      iree_device_size_t byte_length = 0;
      status = desc_byte_length(&outputs[i], &element_type, &byte_length);
      if (!iree_status_is_ok(status)) break;
      /* Remember the host destination and size for the D2H readback. */
      ctx->outputs[i].host_ptr = outputs[i].data;
      ctx->outputs[i].byte_length = byte_length;
    }
  }

  if (!iree_status_is_ok(status)) {
    set_error_status("xtc_iree_setup", status);
    xtc_iree_teardown(ctx);
    return NULL;
  }
  return ctx;
}

static void invoke_impl(xtc_iree_ctx_t *ctx, int readback) {
  iree_runtime_call_reset(&ctx->call);

  iree_status_t status = iree_ok_status();
  for (int i = 0; i < ctx->n_inputs && iree_status_is_ok(status); ++i) {
    status = iree_runtime_call_inputs_push_back_buffer_view(
        &ctx->call, ctx->input_views[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&ctx->call, /*flags=*/0);
  }

  /* Copy outputs D2H only for the readback variant, keeping the copy out of
   * the timed path; the measurement loop uses the plain variant. */
  if (readback) {
    for (int i = 0; i < ctx->n_outputs && iree_status_is_ok(status); ++i) {
      iree_hal_buffer_view_t *out_view = NULL;
      status = iree_runtime_call_outputs_pop_front_buffer_view(&ctx->call,
                                                               &out_view);
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_device_transfer_d2h(
          ctx->device, iree_hal_buffer_view_buffer(out_view),
          /*source_offset=*/0, ctx->outputs[i].host_ptr,
          ctx->outputs[i].byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout());
      iree_hal_buffer_view_release(out_view);
    }
  }

  if (!iree_status_is_ok(status)) {
    set_error_status("xtc_iree_invoke", status);
    fprintf(stderr, "xtc_iree_invoke failed: %s\n", g_error);
    abort();
  }
}

void xtc_iree_invoke(void *handle) {
  invoke_impl((xtc_iree_ctx_t *)handle, /*readback=*/0);
}

void xtc_iree_invoke_readback(void *handle) {
  invoke_impl((xtc_iree_ctx_t *)handle, /*readback=*/1);
}

void xtc_iree_teardown(void *handle) {
  xtc_iree_ctx_t *ctx = (xtc_iree_ctx_t *)handle;
  if (!ctx) return;
  if (ctx->input_views) {
    for (int i = 0; i < ctx->n_inputs; ++i) {
      iree_hal_buffer_view_release(ctx->input_views[i]);
    }
    free(ctx->input_views);
  }
  free(ctx->outputs);
  if (ctx->session) {
    iree_runtime_call_deinitialize(&ctx->call);
    iree_runtime_session_release(ctx->session);
  }
  if (ctx->device) iree_hal_device_release(ctx->device);
  if (ctx->instance) iree_runtime_instance_release(ctx->instance);
  free(ctx);
}
