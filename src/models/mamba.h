#pragma once

#include "models.h"

struct llama_model_mamba : public llama_model_base {
    llama_model_mamba(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
