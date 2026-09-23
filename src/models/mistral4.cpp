#include "models.h"
#include "deepseek2.h"

struct llama_model_mistral4 : public llama_model_deepseek2 {
    llama_model_mistral4(const struct llama_model_params & params) : llama_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_deepseek2

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

LLAMA_MODEL_IMPL(mistral4, "mistral4", llama_model_mistral4)

std::unique_ptr<llm_graph_context> llama_model_mistral4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

