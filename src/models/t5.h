#pragma once

#include "models.h"

struct llama_model_t5 : public llama_model_base {
    llama_model_t5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
