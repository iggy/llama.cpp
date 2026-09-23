#include "models.h"
#include "hunyuan-vl.h"

struct llama_model_hunyuan_dense : public llama_model_hunyuan_vl {
    llama_model_hunyuan_dense(const struct llama_model_params & params) : llama_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_hunyuan_vl

    using graph = llama_model_hunyuan_vl::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

LLAMA_MODEL_IMPL(hunyuan_dense, "hunyuan-dense", llama_model_hunyuan_dense)

std::unique_ptr<llm_graph_context> llama_model_hunyuan_dense::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

