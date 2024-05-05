from langchain_community.llms import LlamaCpp


def load_llms(llm_path):
    n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    agent_llm = LlamaCpp(
        model_path=llm_path,
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        echo=True,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=3500,
        f16_kv=True
    )

    qa_llm = LlamaCpp(
        model_path=llm_path,
        temperature=0.5,
        max_tokens=2000,
        top_p=0.8,
        echo=True,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=3500,
        f16_kv=True
    )

    general_llm = LlamaCpp(
        model_path=llm_path,
        temperature=1,
        max_tokens=2000,
        top_p=1,
        echo=True,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=3500,
        f16_kv=True
    )

    summarizer_llm = LlamaCpp(
        model_path=llm_path,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.3,
        echo=True,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=3500,
        f16_kv=True
    )

    code_llm = LlamaCpp(
        model_path=llm_path,
        temperature=0.2,
        max_tokens=2000,
        top_p=0.5,
        echo=True,
        # callback_manager=callback_manager,
        verbose=False,  # Verbose is required to pass to the callback manager
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=3500,
        f16_kv=True
    )

    return agent_llm, qa_llm, code_llm, general_llm, summarizer_llm