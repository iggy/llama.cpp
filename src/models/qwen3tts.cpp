#include "models.h"
#include "qwen3vl.h"

// llama_model_qwen3tts reuses llama_model_qwen3vl's hparams/tensors/graph logic
struct llama_model_qwen3tts : public llama_model_qwen3vl {
    llama_model_qwen3tts(const struct llama_model_params & params) : llama_model_qwen3vl(params) {}
};

LLAMA_MODEL_IMPL(qwen3tts, "qwen3tts", llama_model_qwen3tts)
