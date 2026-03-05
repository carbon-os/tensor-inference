

// WITHOUT adapter

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





// WITH adapter

root@0040cf722544:~/tensor-inference# ./build/run_qwen_adapter     ~/.cache/models/Qwen/Qwen2.5-0.5B     /root/tensor-adapt/adapters/gomarkdown/step-200     --prompt "show me an example of golang github.com/gomarkdown/markdown"   --max-tokens 4000 
[device]  NVIDIA RTX 2000 Ada Generation  VRAM 15974 MiB
[base]    /root/.cache/models/Qwen/Qwen2.5-0.5B
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
[DEBUG tokenizer] Resolved BOS: 151643, EOS: 151643, PAD: 151643
[model]   qwen2  vocab=151936  layers=24  hidden=896  dtype=bfloat16
[qwen] loading 24 layers
[qwen] all weights loaded
[base]    loaded in 854.656 ms  (942 MiB)
[adapter] /root/tensor-adapt/adapters/gomarkdown/step-200
[adapter] domain:    test/gomarkdown
[adapter] base:      unknown
[adapter] arch:      qwen2
[adapter] rank:      16
[adapter] alpha:     16
[adapter] layers:    0 – 23
[adapter] inject:    Q K V O
[adapter] layer 0 q_proj  A=16×896  B=896×16
[adapter] loaded — scale = 1  (16 / 16)
[adapter] loaded in 6.50646 ms

--- output ---
Here is an example of using golang-markdown with golang-markdown-parser:

```go
import "github.com/omarkdown/markdown"
import "github.com/omarkdown/omarkdown/markdown/ast"
import "github.com/omarkdown/omarkdown/parsers"

m := markdown.NewRenderer()

flag.Set(Override, false) // use engine hooks
renderer := m.Renderer().WithFunc(showFlags) // enter flag hook

renderer.OnDraw = func(dst int) {
   verbosity := omarkdown.EnhancedInterpolationLevel + 1
   if verbosity > 0 {
      flag.Debug(verbosity) // show debug information
   } else {
      os.Write(flag.Output, "\n") // consume out-bound text
   }
}

flag.Set(Override, false) // again override

flag.Render(ast.BlockNode) // write rendered output
}

url := "https://docs.leaflet.io/"
url += "leaflet-ios-header.png" // relative URL

exports := []string{ "separator", "hello" }
data := ast.LiteralNode(ast.NodeString("hello"), exports)

renderer.Render(ast.TextNode, data)

extensions := []string{ "leaflet", "leaflet-ios" }
node := ast.TextNode(ast.NodeString("text"), extensions)

flag.Render(node) // write rendered nodes
<omarkdown.Renderer>
</omarkdown.Renderer>
<omarkdown.Flags> showText | showFlags | hideExtensions | enableEnhancedInterpolation | disablePath/ExpandHierarchy | highlightPackages | pathOfHead
</omarkdown.Flags> // end flag hook
<ol> <li> <p> <p> <p> <p> </p> </p> </li> <p> <p> </p> </p> </li> </ol>
</ul> // write ordered list
</body> <img src="https://docs.leaflet.io/leaflet-ios-header.png" alt="banner"> <a> separator</a> <p> <p> separator</p> <p> separator</p> </p> </a> <p> <p> separator</p> <p> separator</p> </p> </a> </ol> // render unordered list with separator node
</html> // render rich text document
</html> </body> </html>

<ol> <li> <p> <p> separator</p> </p> <p> separator</p> <p> separator</p> </p> <p> separator</p> </p> </li> <li> <p> separator</p> </p> <p> separator</p> <p> separator</p> </li> <li> <p> separator</p> </p> <p> separator</p> <p> separator</p> </p> </li> </ol> // render ordered list with path of headings
</body> </html> </body> </html> </body> </html> </render>`

lo := ast.OrderedListNode(ast.NodeString(flags), data)
flag.Render(lo) // write ordered list with path of headers as separators

var h1 = ast.HeadingNode(flags, "<h1>Header 1</h1>")
flag.Render(h1) // write heading node as raw value

flag.Render(h1, flags) // write heading
<ol> <h1>Header 1</h1> <h1>Header 2</h1> <p> separator</p> <p> separator</p> <p> separator</p> </ol> <h2>Header 3</h2> <p> separator</p> <p> separator</p> <p> separator</p> </ol> </p> </li> </ol> // write unordered list as raw
<ol> <li> <h1>Header 1</h1> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> </body> </body> </html> </renderer><ol> <h1>Header 1</h1> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> </ul> <p> separator</p> </ol> </body> </html> </main>

h1 := ast.HeadingNode(flags, "<h1>Heading 1</h1>")
flag.Render(h1) // render heading

flags.AddExportFlags(omarkdown.JsReferrerPath) // add path of exported file to flag for embedding
<ol> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> <ol> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> </ul> </body> </main></h2> <h2>Header 2</h2> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </ul> </body> </renderer><ul> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> </ol> </ul> </dl><pre>separator</pre> </dl></pre><ul> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> <p> separator</p> </li> <li> <h1>Header 1</h1> <js-embed source="http://flags.leaflet.io" target="_self"> separator</h1> <p> separator</^C
root@0040cf722544:~/tensor-inference# 