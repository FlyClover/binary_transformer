
"""# A Real World Example

> Now we consider a real-world example using the IWSLT German-English Translation task. This task is much smaller than the WMT task considered in the paper, but it illustrates the whole system. We also show how to use multi-gpu processing to make it really fast.
"""

#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de

"""## Data Loading
> We will load the dataset using torchtext and spacy for tokenization.
"""

# """# Additional Components: BPE, Search, Averaging

# > So this mostly covers the transformer model itself. There are four aspects that we didn't cover explicitly. We also have all these additional features implemented in [OpenNMT-py](https://github.com/opennmt/opennmt-py).

# > 1) BPE/ Word-piece: We can use a library to first preprocess the data into subword units. See Rico Sennrich's [subword-nmt](https://github.com/rsennrich/subword-nmt) implementation. These models will transform the training data to look like this:

# ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

# > 2) Shared Embeddings: When using BPE with shared vocabulary we can share the same weight vectors between the source / target / generator. See the [(cite)](https://arxiv.org/abs/1608.05859) for details. To add this to the model simply do this:
# """

# if False:
#     model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
#     model.generator.lut.weight = model.tgt_embed[0].lut.weight

# """> 3) Beam Search: This is a bit too complicated to cover here. See the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py) for a pytorch implementation.

# > 4) Model Averaging: The paper averages the last k checkpoints to create an ensembling effect. We can do this after the fact if we have a bunch of models:
# """

# def average(model, models):
#     "Average models into model"
#     for ps in zip(*[m.params() for m in [model] + models]):
#         p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

# """# Results

# On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
# in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0
# BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is
# listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model
# surpasses all previously published models and ensembles, at a fraction of the training cost of any of
# the competitive models.

# On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
# outperforming all of the previously published single models, at less than 1/4 the training cost of the
# previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
# dropout rate Pdrop = 0.1, instead of 0.3.
# """

# # Image(filename="images/results.png")

# """> The code we have written here is a version of the base model. There are fully trained version of this system available here  [(Example Models)](http://opennmt.net/Models-py/).
# >
# > With the addtional extensions in the last section, the OpenNMT-py replication gets to 26.9 on EN-DE WMT. Here I have loaded in those parameters to our reimplemenation.
# """

# !wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt

# model, SRC, TGT = torch.load("en-de-model.pt")



# model.eval()
# sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
# src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
# src = Variable(src)
# src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
# out = greedy_decode(model, src, src_mask,
#                     max_len=60, start_symbol=TGT.stoi["<s>"])
# print("Translation:", end="\t")
# trans = "<s> "
# for i in range(1, out.size(1)):
#     sym = TGT.itos[out[0, i]]
#     if sym == "</s>": break
#     trans += sym + " "
# print(trans)

# """## Attention Visualization

# > Even with a greedy decoder the translation looks pretty good. We can further visualize it to see what is happening at each layer of the attention
# """

# tgt_sent = trans.split()
# def draw(data, x, y, ax):
#     seaborn.heatmap(data,
#                     xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
#                     cbar=False, ax=ax)

# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     print("Encoder Layer", layer+1)
#     for h in range(4):
#         draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
#             sent, sent if h ==0 else [], ax=axs[h])
#     plt.show()

# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     print("Decoder Self Layer", layer+1)
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
#             tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
#     plt.show()
#     print("Decoder Src Layer", layer+1)
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
#             sent, tgt_sent if h ==0 else [], ax=axs[h])
#     plt.show()

# """# Conclusion

# > Hopefully this code is useful for future research. Please reach out if you have any issues. If you find this code helpful, also check out our other OpenNMT tools.

# ```
# @inproceedings{opennmt,
#   author    = {Guillaume Klein and
#                Yoon Kim and
#                Yuntian Deng and
#                Jean Senellart and
#                Alexander M. Rush},
#   title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
#   booktitle = {Proc. ACL},
#   year      = {2017},
#   url       = {https://doi.org/10.18653/v1/P17-4012},
#   doi       = {10.18653/v1/P17-4012}
# }
# ```

# > Cheers,
# > srush

# {::options parse_block_html="true" /}
# <div id="disqus_thread"></div>
# <script>

# /**
# *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
# *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
# /*
# var disqus_config = function () {
# this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
# this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
# };
# */
# (function() { // DON'T EDIT BELOW THIS LINE
# var d = document, s = d.createElement('script');
# s.src = 'https://harvard-nlp.disqus.com/embed.js';
# s.setAttribute('data-timestamp', +new Date());
# (d.head || d.body).appendChild(s);
# })();
# </script>
# <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

# <div id="disqus_thread"></div>
# <script>
#     /**
#      *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
#      *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
#      */
#     /*
#     var disqus_config = function () {
#         this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
#         this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
#     };
#     */
#     (function() {  // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
#         var d = document, s = d.createElement('script');

#         s.src = 'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace EXAMPLE with your forum shortname!

#         s.setAttribute('data-timestamp', +new Date());
#         (d.head || d.body).appendChild(s);
#     })();
# </script>
# <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>
# """
