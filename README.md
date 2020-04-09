# Learning to Identify High Betweenness Nodes

> Machine Learning with Graphs hw#1 @NCKU

### INTRODUCTION
本次作業為paper [“Learning to Identify High Betweenness Centrality Nodes from 
Scratch: A Novel Graph Neural Network Approach” 
]( https://arxiv.org/abs/1905.10418) 之實作，相較於耗費大量時間計算出一龐大網路中各節點實際的 Betweenness Centrality value，此研究目標在於利用深度學習技術，在encoder-decoder之框架下建構並訓練一個model，預測出各節點之間相對應的BC值大小，進而有效且快速地找出在龐大網路中BC值較高之重要節點。

此報告主要分為兩大部分，先是藉由[此程式碼](https://github.com/emschenn/mlg_hw1/blob/master/DrBC.ipynb)，在實作方法上做詳細的解說；而後再針對此實作計算出的結果做比較和分析。





### CODE IMPLEMENTATION
#### **STEP 1 - 創建model**
![](https://i.imgur.com/letoC2p.png)

根據上圖，細分為encoder/max pooling/decoder來建立model：

**1.1 Encoder**

encoder負責生成各點之feature，首先將input [deg(v), 1, 1] 經由fully-connected layer和ReLU轉成128維，之後的處理順序分別為： 


 
1.	Neighborhood Aggregation: GCN

為了省去直接計算兩點間最短距離之成本，這裡選擇使用Graph Convolutional來對圖做局部的aggregation操作，實作上直接套用torch_geometric中的GCNConv()


```python=
# prepare model
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  
        self.lin = torch.nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        return aggr_out
```



2.	Combine: GRU

為了取得更佳的feature，我們需要對這層此node之neighborhood 的embedding 與此node自己上一層的embedding做Combine，而根據論文，這裡選用相較其他method而言，更具彈性的GRU來達成目標！

3.	Layer Aggregation: max pooling

結束5次的Aggregation和Combine後，得到5個128維的feature，再從中以element-wise的方式，取出5個值中的最大值，得到一個128維的output。

```python=  
        l = [x1[0],x2[0],x3[0],x4[0],x5[0]]
        l = torch.stack(l)
        x = torch.max(l, dim=0).values
        return x
```

**1.2 Decoder**

Decoder再負責將各點之feature轉成預測之BC value。而根據論文，這裡選擇ReLu作為我們的activation function。

```python=  
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 1)
```

最後得出


#### **STEP 2 - 準備train data**

利用networkx之powerlaw_cluster_graph (n, m, p, seed=None) 生成n（在此model中設batch_size=16）個網路，其中參數之設置：

* n- num of nodes : random(150,200)
* m- num of random edges to add each new node: 4
* p- probability of adding a triangle after adding a random edge: 0.05

生成n個網路後，整理資料求出：

* deg_list：(|V| x 3) = [[deg(n0), 1, 1], [deg(n1), 1, 1], [deg(n2), 1, 1], …]
* edge_index：(2 x |E|) = [[S], [T]]  { for all edge(s,v) in G | s∈S, t∈T }


利用networkx之betweenness_centrality(n) 算出各點實際的BC值，這裡特別要注意的是為了避免BC值太小（都趨近於0），model train不起來，因此算出之BC值得再做log處理，使彼此之差值變大，結果才可成功收斂。

* bc_value：(|V| x 1) = [bc(n0), bc(n1), bc(n2), …] 

最後，用來比較兩兩node之間實際bc值的差值和predict之bc相對值的差值，採用論文的作法：randomly sample 5|V | source nodes and 5|V | target nodes來做比較之pair。

* pair_index：(2 x 5|V|) = [[shuffle(node_in_G*5)], [shuffle(node_in_G*5)]] 


#### **STEP 3 - 開始train!**

開始train！其中Optimizer、參數之設置皆參考於論文，training過程如下：

* learning rate: 0.0001
* batch size: 16
* iteration: 10000

最後之loss平均結果約落在：6000-7000上下


#### **STEP 4 - 驗證訓練結果**

**4.1 Read File**

讀檔，將資料放進model得到結果

```python=
  f = readFile('y') #y for com_youtube,num for synthetic data
  f.get_deg_list() 
  f.get_edge_index()  
  f.get_bc_value()  
  
  outs = model(f.get_deg_list(),f.get_edge_index())
```
**4.2 Top-N % accuracy**

將經model運算得出的BC相對值以及實際計算得到BC值做排序，取出兩者前N%，再比較其相同node的數量作為一個準確度的判斷：

```python=
def takeSecond(elem):
    return elem[1]
    
def topN_accuracy(file,outs,n):
  predict_value,bc_value = [],[]
  for i,j in enumerate(outs.tolist()):
    predict_value.append([i,*j])
  bc_value = f.get_bc_value()
  bc_value.sort(key = takeSecond,reverse = True)
  predict_value.sort(key = takeSecond,reverse = True)
  p,t = [],[]
  for x in range(int(len(predict_value)*n/100)):
    p.append(predict_value[x][0])
    t.append(bc_value[x][0])
  return(len(set(t)&set(p)) / len(p))
```

**4.3 Kendall tau**

利用Kendall tau直接比較實際值/預測值間排序的相似程度：

```python=
import scipy.stats as stats
def kendall_tau(file,outs):
  predict_value,bc_value = [],[]
  for i,j in enumerate(outs.tolist()):
    predict_value.append(*j)
  for i in file.get_bc_value():
    bc_value.append(i[1])
  # print(predict_value)
  # print(bc_value)
  tau, _ = stats.kendalltau(predict_value, bc_value)
  return(tau)
```




### EXPERIMENTAL RESULTS

#### 結果1 – 根據論文之實作

利用30筆生成網路之data/ 真實網路youtube之data驗證此model：

| data	| Top-1%	| Top-5%| 	Top-10%	| Kendal| 
| -------- | -------- | -------- |-------- |-------- |
 |0	 |0.98	 |0.892	 |0.85	 |0.632408 |
 |1	 |0.94	 |0.924	 |0.856	 |0.628225 | 
 |2	 |0.92	 |0.892	 |0.856	 |0.640236 |
 |3	 |0.96	 |0.884	 |0.854	 |0.630935 |
 |4	 |0.96	 |0.872	 |0.876	 |0.636721 |
 |5	 |0.92	 |0.888	 |0.85	 |0.637008 |
 |6	 |0.9	 |0.9	 |0.85	 |0.63165 |
 |7	 |0.96	 |0.908	 |0.858	 |0.640853 |
 |8	 |0.9	 |0.864	 |0.886	 |0.635994 |
 |9	 |0.96	 |0.896	 |0.886	 |0.630207 |
 |10	 |0.94	 |0.9	 |0.868	 |0.629056 |
 |11     |0.94	 |0.876	 |0.842  |0.634303 |
 |12	 |0.9	 |0.86	 |0.86	 |0.635016 |
 |13	 |0.9	 |0.892	 |0.856	 |0.635277 |
 |14	 |0.94	 |0.856	 |0.864	 |0.634768 |
 |15	 |0.96	 |0.888	 |0.874	 |0.639659 |
 |16	 |0.92	 |0.868	 |0.878	 |0.62671 |
 |17	 |0.94	 |0.888	 |0.85	 |0.633372 |
 |18	 |0.98	 |0.908	 |0.848	 |0.633991 |
 |19	 |0.96	 |0.884	 |0.862	 |0.640018 |
 |20	 |0.96	 |0.892	 |0.878	 |0.635776 |
 |21	 |0.96	 |0.872	 |0.86 |	0.641155 |
 |22	 |0.96	 |0.888 |	0.87 |	0.640104 |
 |23	 |0.92	 |0.876	 |0.87 |	0.636818 |
 |24	 |0.92	 |0.9	 |0.876	 |0.637847 |
 |25 |	0.94	 |0.892	 |0.874	 |0.633936 |
 |26	 |0.96	 |0.9	 |0.866	 |0.633009 |
 |27 |0.98	 |0.884 |	0.868 |	0.63469 |
 |28	 |0.94	 |0.904	 |0.85	 |0.62656 |
 |29 |	0.96	 |0.908	 |0.866	 |0.633401 |
 |Avg of 30 Synthetic Network	 |0.942666667	 |0.888533333	 |0.8634	 |0.634656822 |
 |Youtube  |	0.62663	 |0.59800 |	0.6234 |	0.3975 |

根據結果可以發現此model之表現明顯在生成網路上是更勝於真實網路的，測試了30筆具5000個節點之不同的生成網路，標準差平均約0.017，top-10% rank以內的準確率普遍到達85%以上，Kendal tau係數平均也有0.63；相較於真實網路，top-10% rank以內的準確率平均只有約6成，Kendal tau係數也降到0.4。
然而會有這樣的結果，也可能是因為兩個網路的節點數相差太大（生成網路只有5000個node，然而youtube資料有10幾萬個），這部分也是有待討論。

#### 結果2 – 不同training data

**training data: powerlaw_cluster_graph(n=random(150,200), m=4, p=0.05, seed=None)**

針對n, m, p三組參數做實驗：

*	n- 生成網路中的node數

||50-100||	150-200	||400-500||
| -------- | --------- | --------- | --------- | --------- |  --------- | ------- |
||	Synthetic	|Youtube	|Synthetic	|Youtube|	Synthetic|	Youtube|
|Top-1%	|0.94	|0.581	|0.94|	0.627|	0.94	|0.6458|
|Top-5%	|0.924|	0.5553	|0.924|	0.598	|0.924	|0.62323|
|Top-10%	|0.856	|0.5933	|0.856|	0.623	|0.856	|0.6465|
|Kendal	|0.6268|	0.0025	|0.628|	0.397	|0.6106	|0.23428|

根據結果可以發現training data中網路的node數對於較小的生成網路來說影響不大，而對於龐大的真實網路來說，準確率似乎有提升一點點。


*	m- 生成網路中each node的基本邊數

||m = 2	||m = 4||	m = 6||
| -------- | --------- | --------- | --------- | --------- |  --------- | ------- |
||Synthetic|	Youtube|	Synthetic|	Youtube|	Synthetic|	Youtube|
|Top-1%|	0.94|	0.476|	0.94	|0.627|	0.94|	0.411||
|Top-5%	|0.924	|0.519	|0.924|	0.598|	0.924	|0.523|
|Top-10%|	0.856	|0.584	|0.856|	0.623	|0.864|	0.536|
|Kendal|	0.628|	0.16	|0.628	|0.397	|0.607|	0.213|


根據結果可以發現training data中網路的邊數同樣對於較小的生成網路來說影響不大，但對於較龐大的真實網路來說，論文中實作的邊數4的確是最好的結果。


  
*	p- 生成網路中新邊形成三角形的機率

|	|p = 0.01	||p = 0.05	||p = 0.1||
| -------- | --------- | --------- | --------- | --------- |  --------- | ------- |
||Synthetic	|Youtube|	Synthetic	|Youtube	|Synthetic	|Youtube|
|Top-1%	|0.94	|0.3295|0.94	|0.627|	0.94|	0.5386|
|Top-5%	|0.924	|0.5468	|0.924	|0.598	|0.924|	0.5686|
|Top-10%|	0.855	|0.604|0.856	|0.623	|0.864	|0.5004|
|Kendal	|0.628	|0.393	|0.628	|0.397	|0.607|	-0.031|

根據結果同樣可以發現training data中三角形的數目越多，對於較小的生成網路來說影響不大，但對於較龐大的真實網路來說，雖然Top-10%以內的準確率看似沒有相差太多，但Kendal tau係數卻成了負相關，可見training data之網路選擇也不可過於群聚。



#### 結果3 – 不同dimension 

測試不同embedding dimension對於結果的影響：

|           | dim = 64 |           | dim = 128 |           | dim = 256    |         |
| -------- | --------- | --------- | --------- | --------- |  --------- | ------- |
|          | Synthetic | Youtube   | Synthetic | Youtube   |      Synthetic | Youtube |
| Top-1%   |   0.94    | 0.5195    | 0.94      | 0.627     |      0.92      | 0.024   |
| Top-5%   |	0.924  |0.5506     |	0.924|	0.598	|0.92	|0.272|
| Top-10%  |	0.856  |0.5941     |	0.856|	0.623	|0.856	|0.483|
| Kendal   |	0.6247 |0.3485     |	0.628|	0.397	|0.6264	|0.413|


根據結果可以發現，dim的調整對於較小的生成網路影響皆不大，而在youtube網路上，dim越低的確結果越差，然而dim越高，雖然Kendal tau係數變大了，但top-K之準確率卻相當的低。對於此現象，推測有可能是因為training data的網路較小，所以測試較龐大的網路時產生overfitting。




