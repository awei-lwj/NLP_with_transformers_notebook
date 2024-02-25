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