#pragma once

#include "models.h"

struct llama_model_nemotron_h : public llama_model_base {
    llama_model_nemotron_h(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
            const llama_model & model, int64_t n_embd_head, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
