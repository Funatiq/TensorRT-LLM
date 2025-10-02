(additional-outputs)=


# Additional Outputs

TensorRT-LLM provides several options to return additional outputs from the model during inference. These options can be specified in the `SamplingParams` object and control what extra information is returned for each generated sequence.

## Options

### `return_context_logits`
- **Description**: If set to `True`, the logits (raw model outputs before softmax) for the context (input prompt) tokens are returned for each sequence.
- **Usage**: Useful for tasks such as scoring the likelihood of the input prompt or for advanced post-processing.
- **Default**: `False`

### `return_generation_logits`
- **Description**: If set to `True`, the logits for the generated tokens (tokens produced during generation) are returned for each sequence.
- **Usage**: Enables advanced sampling, custom decoding, or analysis of the model's output probabilities for generated tokens.
- **Default**: `False`

### `logprobs`
- **Description**: If set to an integer value `N`, the top-`N` log probabilities for each generated token are returned, along with the corresponding token IDs.
- **Usage**: Useful for uncertainty estimation, sampling analysis, or for applications that require access to the probability distribution over tokens at each generation step.
- **Default**: `None` (no log probabilities returned)

### `additional_model_outputs`
- **Description**: A list of `AdditionalModelOutput` objects specifying extra tensors to be returned by the model for each sequence. Each `AdditionalModelOutput` defines the name of the output and whether it should be gathered for the context (input prompt) or for the generated tokens.
- **Usage**: Enables retrieval of additionaal model states or intermediate results, such as hidden states, or any other supported output, for advanced analysis, debugging, or research purposes.
- **How to use**:
    - Create one or more `AdditionalModelOutput` objects, specifying the `name` (must match a supported output) and `gather_context` (set to `True` to collect outputs for the input prompt, or `False` for generated tokens).
    - Pass the list of these objects to the `additional_model_outputs` parameter of `SamplingParams`.
    - After generation, access the results via `sequence.additional_context_outputs` (for context outputs) and `sequence.additional_generation_outputs` (for generation outputs).
- **Default**: `None` (no additional outputs returned)

**Note:** The available output names depend on the model implementation. The model forward function is expected to return a dictionary of model outputs including the `"logits"` and any additional output that should be attached to responses.
