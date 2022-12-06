### Dataset and Base Model

we use the script mentioned above to download preprocessed OPUS multi-domain De-En dataset
and pretrained WMT19 De-En winner model. If you are interested in the general domain dataset for training pretrained models, download it from [here](https://www.statmt.org/wmt19/translation-task.html).

### Datastore Size

<table>
  <tr>
    <th align="center">Domain</th>
    <td align="center">IT</td>
    <td align="center">Medical</td>
    <td align="center">Koran</td>
    <td align="center">Law</td>
  </tr>
  <tr>
    <th align="center">Size</th>
    <td align="center">3602862</td>
    <td align="center">6903141</td>
    <td align="center">524374</td>
    <td align="center">19061382</td>
  </tr>
</table>

### BLEU Score

|model \ domain|IT|Medical|Koran|Law|General Domain (Avg)|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Base NMT|38.35|40.06|16.26|45.48|37.16|
|vanilla kNN-MT|45.88|54.13|20.51|61.09|11.98|
|adaptive kNN-MT|47.41|56.15|20.20|63.01|32.70|
|kernel smoothed kNN-MT|47.81|56.67|20.16|63.27|32.60|

> Note that due to the unavoidable randomness in constructing the faiss index, there will be slight differences between our results and the paper's, and our results will not be exactly the same as yours.

- hyper-parameter of vanilla kNN-MT

  we follow the configure of [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt)

  |domain|IT|Medical|Koran|Law|
  |:--:|:--:|:--:|:--:|:--:|
  |K|8|4|16|4|
  |Temperature|10|10|100|10|
  |Lambda|0.7|0.8|0.8|0.8|

- hyper-parameter of adaptive kNN-MT

  we choose the max-k which obtained highest BLEU for every domain in 
  [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt)

  <table>
    <tr>
      <th align="center">Domain</th>
      <td align="center">IT</td>
      <td align="center">Medical</td>
      <td align="center">Koran</td>
      <td align="center">Law</td>
    </tr>
    <tr>
      <th align="center">max-k</th>
      <td align="center">8</td>
      <td align="center">16</td>
      <td align="center">16</td>
      <td align="center">8</td>
    </tr>
  </table>

  although `temperature` is trainable, we follow the paper and fix `temperature` to 10.  

- hyper-parameter of kernel smoothed kNN-MT

  for all domains, we use fixed `k` 16.