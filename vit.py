import math
import torch
from torch import clone, nn

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        input = clone(input)
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    #Convert the image into patches and then projection 

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    #Combine the patch embeddings with the class token and position embeddings.

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        #making a token that can be added to input sequence and used to classify
        self.classify_t = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        #create position embeddings for the token and patch embeddings and adding 1 to sequence length for the token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.Dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        classify_ts = self.classify_t.expand(batch_size, -1, -1)

        x = torch.cat((classify_ts, x), dim=1)
        x = x + self.position_embeddings
        x = self.Dropout(x)
        return x

class AttentionHead(nn.Module):
    #single attention head
    #multiple of these are used in multihead attention

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        #creating query, key and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #projecting the input in query, key and value
        #then using the same to generate the query, value, and key
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        #attention scores
        #softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        #calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class MultiHeadAttention(nn.Module):
    #multi head attention
    #this module is used in Transformer encode module

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        #calculation attention head size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #to use bias or not in projections
        self.qkv_bias = config["qkv_bias"]
        
        #Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(self.hidden_size, self.attention_head_size, config["attention_probs_dropout_prob"], self.qkv_bias)
            self.heads.append(head)
            
        #Creating a linear layer to project the attention output back to hidden size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        #Calculation attention for each attention head
        attention_outputs = [head(x) for head in self.heads]
        #Concatenate the attention outputs from each attention heads
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        #Projecting the concatenated attention ouput back to hidden_size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        #Return the attention output and the attention probabalities
        if not output_attentions:
            return(attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)

class MLP(nn.Module):
   # Multilayer perceptron module
   
   def __init__(self, config):
      super().__init__()
      self.dense1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
      self.activation = NewGELUActivation()
      self.dense2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
      self.dropout = nn.Dropout(config["hidden_dropout_prob"])
      
   def forward(self, x):
      x = self.dense1(x)
      x = self.activation(x)
      x = self.dense2(x)
      x = self.dropout(x)
      return x

class Block(nn.Module):
   #a single transformer block
   
   def __init__(self, config):
      super().__init__()
      self.attention = MultiHeadAttention(config)
      self.layernorm1 = nn.LayerNorm(config["hidden_size"])
      self.mlp = MLP(config)
      self.layernorm2 = nn.LayerNorm(config["hidden_size"])
      
   def forward(self, x, output_attentions=False):
      #Self attention
      attention_output, attention_probs = \
         self.attention(self.layernorm1(x), output_attentions=output_attentions)
      #skip connection
      x = x + attention_output
      #Feed forward network
      mlp_output = self.mlp(self.layernorm2(x))
      #skip connection
      x = x + mlp_output
      #Returning the transformer's block output and the attention probabilities
      if not output_attentions:
         return(x, None)
      else:
         return(x, attention_probs)

class Encoder(nn.Module):
   # transformer encoder module
   
   def __init__(self, config):
      super().__init__()
      #Creating a transformer block
      self.blocks = nn.ModuleList([])
      for _ in range(config["num_hidden_layers"]):
         block = Block(config)
         self.blocks.append(block)
         
   def forward(self, x, output_attentions=False):
      #Caculate the transformer block's output for each block
      all_attentions = []
      for block  in self.blocks:
         x, attention_probs = block(x, output_attentions=output_attentions)
         if output_attentions:
            all_attentions.append(attention_probs)
      #Return encoder's output and the attention probabilities
      if not output_attentions:
         return(x, None)
      else:
         return(x, all_attentions)

class ViTForClassification(nn.Module):
   #the Vision transformer for classfication
   
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.image_size = config["image_size"]
      self.hidden_size = config["hidden_size"]
      self.num_classes = config["num_classes"]
      #Create embedding module
      self.embedding = Embeddings(config)
      #Create the transformer encoder module
      self.encoder = Encoder(config)
      #Create a linear layer to project the encoder's output to the number of classes
      self.classifier = nn.Linear(self.hidden_size, self.num_classes)
      #Initialize the weights
      self.apply(self._init_weights)
      
   def forward(self, x, output_attentions=False):
      #Calculate the embedding output
      embedding_output = self.embedding(x)
      #Calculate the encoder's output
      encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
      #Calculate the logits, taking the Classify token's output as feature for classfication
      logits = self.classifier(encoder_output[:, 0])
      #Return the logits and the attention probabailities
      if not output_attentions:
         return(logits, None)
      else:
         return(logits, all_attentions)

   def _init_weights(self, module):
      if isinstance(module, (nn.Linear, nn.Conv2d)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
      elif isinstance(module, Embeddings):
        module.position_embeddings.data = nn.init.trunc_normal_(
             module.position_embeddings.data.to(torch.float32),
             mean=0.0,
             std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

        module.classify_t.data = nn.init.trunc_normal_(
             module.classify_t.data.to(torch.float32),
             mean=0.0,
             std=self.config["initializer_range"],
            ).to(module.classify_t.dtype)
