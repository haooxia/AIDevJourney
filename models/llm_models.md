# LLM Models Overview

- [LLM Models Overview](#llm-models-overview)
  - [Basic Knowledge](#basic-knowledge)
    - [LLM Inference Process](#llm-inference-process)
      - [prefill phase](#prefill-phase)
      - [decode phase](#decode-phase)
      - [related metrics](#related-metrics)
  - [gemma-3n](#gemma-3n)
  - [qwen3](#qwen3)


## Basic Knowledge

### LLM Inference Process

#### prefill phase

**prefill** phase(预填充阶段):

* 简单定义：用户输入完prompt到生成第一个token的过程。
* 严格定义：该阶段是对输入prompt进行一次完整的、并行的forward propagation，为后续decode阶段生成初始的上下文状态。
* 模型对n个token进行self-attention计算，n个token每一个都会与自己以及其他n-1个token计算attention，得到一个n*n的attention matrix。由于n个token固定，故而该计算过程可以**高度并行**。
* 注意：计算中，每个transformer层的每个attention head都会为n个token生成对应的key向量和v向量，然后会存到**KV Cache**高速缓存中。

**prefill vs. encoder of Transformer**

* prefill并不是transformer的encoder的工作
* 我们知道gpt是decoder-only，所以它训练与推理中的确没有transformer的encoder。
* prefill是将prompt输入decoder-only Transformer的所有层，本质上是decoder在理解上下文（虽然和transformer中并行的encoder很像）
* 然后我们一般说transformer中encoder支持并行，~~decoder由于mask机制不支持~~
  * 上述错误：decoder的**mask是不让当前token看到后续的token**，但是我是同时知道n个输入的token的，每个token确实看不到后续，但可以看到左侧，即使信息有限，但**不影响每个位置是可以并行**这件事情。(也即注意力矩阵上三角部分单独置零)
  


#### decode phase

**decode**阶段(解码):
* 简单定义：迭代地、**自回归地**逐词生成输出内容。
* 在每一步中，模型接收前一步生成的词元作为输入，利用 Prefill 阶段和之前 Decode 步骤中生成的 KV Cache，来高效地计算并生成下一个词元。
  * prefill在计算并填充初始的KV cache, decode在增量式地使用和更新KV cache

#### related metrics

prefill:

* **TTFT: Time To First Token**, 即从用户输入prompt到生成第一个token的时间
  * prefill latency (ms): TTFT的另一种说法吧
  * ttft越长，用户等待时间越长，体验越差
* prefill speed/rate (tokens/sec): 每秒能处理多少个输入token

decode:
* decode speed (tokens/sec): 每秒能生成多少个输出token
* TPS(TPOT): Tokens Per Second / Tokens Per Output Time，即模型生成token的速度
  * 跟decode speed一样吧

---

benchmark_prefill_tokens: 
> 模拟输入时上下文token数量/长度

1. 当token太少时候，GPU/NPU吃不饱，prefill speed慢
2. 当token太多时候，计算和内存访问负载加大，prefill speed也可能下降
3. 故而prefill speed往往会随token先上升后持平甚至下降，有个最佳区间
4. （该条往往几乎无影响）此外，增加prefill_tokens会降低decode speed，因为decode会从显存中加载KV Cache, 而prefill_tokens越大，KVCache越大，因此decode需要更多时间去读取更多数据。

benchmark_decode_tokens:

1. tokens越多，decode speed一般不受影响，因为decode是自回归嘛，然后token越多，测试结果越稳定。
1. 对prefill speed没影响，因为prefill在decode之前完成

## gemma-3n

Gemma 3n models are designed for efficient execution on everyday devices such as laptops, tablets or phones.

* gemma-3n基于140多种口语的数据训练而成
* 使用selective parameter activation技术来减少资源需求，This technique allows the models to operate at an effective size of 2B and 4B parameters, which is lower than the total number of parameters they contain.
* ollama上有两个model可供使用：
  * gemma-3n-e2b
  * gemma-3n-e4b

> 底层可以参考paper: MatFormer: Nested Transformer for Elastic Inference
> 参考: https://ai.google.dev/gemma/docs/gemma-3n?authuser=4


E4B模型内嵌E2B模型，当question easy时候，会自动使用E2B快速得到模型，当hard时候，会自动使用E4B模型来得到deep analysis

> Gemma3 is different from Gemma3n

## qwen3

qwen3是Qwen系列中最新一代的llm，特性如下：
* 全尺寸稠密特性comprehensive suite of dense：提供0.6b, 1.7b, 4b, 8b, 14b, 30b, 32b, 235b版本
  * Dense model是每次推理会激活模型的所有参数
* 混合专家模型mixture-of-experts (MoE)特性: MoE核心思想是并非每次都用全部模型参数，而是根据输入选择部分“专家”模型来处理，以节省资源。
  * 每次处理输入时，一个router/路由器根据input，选择其中少数几个专家(e.g., 2)来处理，输出是选中专家的输出加权。