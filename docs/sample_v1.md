


root@host:~/tensor-inference# ./build/run_qwen ~/.cache/models/Qwen/Qwen2.5-0.5B     --prompt "show me an example of golang github.com/gomarkdown/markdown package"     --max-toke
ns 8000     --temperature 0.8
[device] NVIDIA RTX 2000 Ada Generation  VRAM 15974 MiB
[load]  model dir: /root/.cache/models/Qwen/Qwen2.5-0.5B
[DEBUG config] /root/.cache/models/Qwen/Qwen2.5-0.5B/config.json:
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}
[DEBUG tokenizer] Opening: /root/.cache/models/Qwen/Qwen2.5-0.5B/tokenizer.json
[DEBUG tokenizer] Loaded vocab size: 151643
[DEBUG tokenizer] Loaded 151387 merges (151387 strings, 0 arrays).
[DEBUG tokenizer] Loaded 22 added_tokens.
[DEBUG tokenizer] Opening config: /root/.cache/models/Qwen/Qwen2.5-0.5B/tokenizer_config.json
[DEBUG tokenizer] Resolved BOS: -1, EOS: -1, PAD: -1
[model] qwen2  vocab=151936  layers=24  hidden=896  dtype=bfloat16
[qwen] loading 24 layers
[qwen] all weights loaded
[load]  322.068 ms  (942 MiB)

--- output ---
show me an example of golang github.com/gomarkdown/markdown package.markdown.ashx

To use the Golang Markdown package with the GitHub.com/GoMarkdown/Markdown/ashx, you can follow these steps:

1. Clone the GitHub repository:
```bash
git clone https://github.com/your-repo/Markdown.git
cd Markdown
```
2. Install the required dependencies:
```bash
go get github.com/your-repo/Markdown
go get github.com/your-repo/Markdown/ashx
```
3. Create a new Markdown file with a .md extension and add some text:

```md
Hello, world!
This is a markdown file.
```

4. Create an HTML file with a .html extension and the same text:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Hello, world!</title>
</head>
<body>
    <p>This is a markdown file.</p>
</body>
</html>
```

5. Open the HTML file in a web browser and view the markdown file.

That's it! You've successfully created a markdown file with Golang Markdown using the GitHub.com/GoMarkdown/Markdown/ashx.
--- stats ---
  tokens:  236
  speed:   10.378 tok/s
  reason:  eos
root@host:~/tensor-inference# 