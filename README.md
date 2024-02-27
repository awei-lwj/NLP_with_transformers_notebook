看是认真看了，只是chapter02写的一般







# Error

Chapter02 code error

Chapter03 text error

我们很快就会看到每个组件的细节，但我们已经可以在图3-1中看到一些描述Transformer架构的东西：

使用我们在第2章中遇到的技术，将输入的文本标记化并转换为标记嵌入。由于注意力机制不知道标记的相对位置，我们需要一种方法将一些关于标记位置的信息注入输入，以模拟文本的顺序性。**因此 因此**，标记嵌入与包含每个标记的位置信息的位置嵌入相结合。
编码器由一叠编码器层或 "块 "组成，这类似于计算机视觉中的卷积层的堆叠。解码器也是如此，它有自己的解码器层堆叠。
编码器的输出被送入每个解码层，然后解码器产生对序列中最可能的下一个符号的预测。这一步的输出再被反馈到解码器，以生成下一个标记，如此反复，直到达到一个特殊的序列结束（EOS）的标记。在图3-1的例子中，设想解码器已经预测了 "Die "和 "Zeit"。现在它得到了这两个词的输入，以及所有编码器的输出来预测下一个标记 "fliegt"。在下一个步骤中，解码器得到 "fliegt "作为额外的输入。我们重复这个过程，直到解码器预测到EOS标记或我们达到最大长度。

Chapter03 coder error 第一个错误
```python
from transformers import AutoTokenizer 
from bertviz.transformers_neuron_view import BertModel 
from bertviz.neuron_view import show 
model_ckpt = "bert-base-uncased" tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt) 
text = "time flies like an arrow" 
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)
```
改成
```python
!pip install bertviz
from transformers import AutoTokenizer 
from bertviz.transformers_neuron_view import BertModel 
from bertviz.neuron_view import show 
model_ckpt = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt) 
text = "time flies like an arrow" 
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

```
```python
import torch from math 
import sqrt 
query = key = value = inputs_embeds 
dim_k = key.size(-1) 
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k) 
scores.size() 

torch.Size([1, 5, 5])


```
```python
import torch 
from math import sqrt 
query = key = value = inputs_embeds 
dim_k = key.size(-1) 
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k) 
scores.size()
```

chapter05

多了一个pd.DataFrame(iterations)

```python
import pandas as pd
input_txt = "Transformers are the" 
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device) 
iterations = [] 
n_steps = 8 
choices_per_step = 5
with torch.no_grad(): 
	for _ in range(n_steps): 
		iteration = dict() 
		iteration["Input"] = tokenizer.decode(input_ids[0]) 
		output = model(input_ids=input_ids) 
		# Select logits of the first batch and the last token and apply softmax 			
		next_token_logits = output.logits[0, -1, :]
		next_token_probs = torch.softmax(next_token_logits, dim=-1) 
		sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True) 
		# Store tokens with highest probabilities 
		for choice_idx in range(choices_per_step): 
			token_id = sorted_ids[choice_idx] 
			token_prob = next_token_probs[token_id].cpu().numpy() 
			token_choice = ( f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)" ) 
			iteration[f"Choice {choice_idx+1}"] = token_choice 
			# Append predicted next token to input 
			input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1) 
			iterations.append(iteration) #pd.DataFrame(iterations)
pd.DataFrame(iterations)


```

文本重复

我们还可以看到贪婪搜索解码的一个主要缺点：往往会产生重复的输出序列，这在新闻报道中当然不可取。我们还可以看到贪婪搜索解码的一个主要缺点：它倾向于产生重复的输出序列，这在一篇新闻文章中当然是不可取的。


Chapter 06
Top-K 抽样那里，文本重复

然后是急剧下降，只有少数几个概率在10-2和10-1之间的标记出现。看这张图，我们可以看到，选择概率最高的标记的 挑选概率最高的标记（10-1处的孤立条）的概率是1/10

文本重复2

k的值是手动选择的，对序列中的每个选择都是一样的，与实际的输出分布无关。序列中的每个选择都是一样的，与实际的输出分布无关。

chaper06 code

```python
from datasets import load_dataset 
dataset = load_dataset("cnn_dailymail", version="3.0.0")
print(f"Features: {dataset['train'].column_names}") 

```

to

```python
from datasets import load_dataset

try:
  dataset = load_dataset("cnn_dailymail","3.0.0")
  print(f"Features: {dataset['train'].column_names}")
except Exception as e:
  print(f"An error occurred: {e}")


```


```python
est_sampled = dataset["test"].shuffle(seed=42).select(range(1000)) 
score = evaluate_summaries_baseline(test_sampled, rouge_metric) 
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names) pd.DataFrame.from_dict(rouge_dict, orient="index", columns=["baseline"]).T


```

少了一个t > 'est_sampled = dataset["test"].shuffle(seed=42).select(range(1000)) '


chapter 08
翻译错误

<这里我们还指定了我们的模型应该期望的类的数量。然后我们可以把这个配置提供给AutoModelForSequenceClassification类的from_pretrained()函数，如下所示：>应该是标签数量


<第八章的优化部分两个动态裁剪完全看不懂>


Chapter09 要掉一个API

Chapter10 colab用不了那个数据集一直在爆disk，下不下去那个数据集算了

chapter11 鉴鉴于像GPT-3这样的大型语言模型估计要花费460万美元来训练。 打错了
而且这个是只有文字介绍
