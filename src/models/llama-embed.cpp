#include "models.h"
#include "llama.h"

struct llama_model_llama_embed : public llama_model_llama {
    llama_model_llama_embed(const struct llama_model_params & params) : llama_model_llama(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_llama

    template <bool embed>
    using graph = llama_model_llama::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

LLAMA_MODEL_IMPL(llama_embed, "llama-embed", llama_model_llama_embed)

std::unique_ptr<llm_graph_context> llama_model_llama_embed::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<true>>(*this, params);
}

