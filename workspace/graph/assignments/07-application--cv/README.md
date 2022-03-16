# Deep Learning on Graphs, 06 Application in COmputer Vision (CV)

---

## Environment Setup

The virtual environment used is defined by [this Conda environment file](docker/environment/graph/graph.yaml)

---

## Up & Running

First, run the command below inside Docker:

```bash
jupyter-lab --ip=0.0.0.0 --port=6006 --no-browser --allow-root
```

Then the Jupyter notebook will be available at **localhost:46006**

---

## Questions

1. 图的自编码器和图卷积神经网络的区别是什么？

    **ANS**

    * **图自编码器**是图数据上的一种无监督学习方法, 可以使用图卷积神经网络构建图自编码器, 也可以使用其他的图神经网络构建图自编码器
    * **图卷积神经网络**是图数据上进行特征变换的手段之一, 算法类似图像数据上的卷积神经网络, 但是基于邻域与Message Passing定义卷积算子

2. 图的变分自编码器比起图的自编码器的优点是什么？

    **ANS**

    * 所获得的embedding为一个多元高斯分布
    * 在进行decoding时, sampling等效为一种regulation, 使用得当，能使得获得的模型拥有更好的泛化能力
    * Variational GAE是一个生成模型, 可以通过在embedded Gaussian上采样, 获得新的特征表示

---

## Classification Pipeline

---

### Dataset Generation

---

### Auto-Encoders

#### GAE

```Python
class GCNEncoder(torch.nn.Module):
    """ deep GCN encoder
    """
    def __init__(self, in_feats, out_feats):
        """ init layers
        """
        super().__init__()
        
        self.gcn1 = GraphConv(
            in_feats=in_feats, out_feats=2*out_feats, 
            weight=True, bias=True, 
            activation=F.relu, 
            allow_zero_in_degree=True
        )
        
        self.output = GraphConv(
            in_feats=2*out_feats, out_feats=out_feats, 
            weight=True, bias=True, 
            activation=None, 
            allow_zero_in_degree=True
        )
    
    def forward(self, g, x):
        """ forward propagation
        """
        h = self.gcn1(g, x)
        h = self.output(g, h)
        
        return h
    
class InnerProductDecoder(torch.nn.Module):
    """ inner product decoder
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, g, z, sigmoid=True):
        """ forward propagation
        """
        g.ndata['z'] = z
        g.apply_edges(fn.u_dot_v('z', 'z', 'logit'))
        logit = g.edata['logit'].sum(dim=1)
        
        return torch.sigmoid(logit) if sigmoid else logit
    
class GAE(torch.nn.Module):
    """ graph autoencoder
    """
    EPSILON = 1e-16
    
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, *args, **kwargs):
        """ encode
        """
        return self.encoder(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """ decode
        """
        return self.decoder(*args, **kwargs)
    
    def get_reconstruction_loss(self, z, pos_g, neg_g, pos_edge_idx, neg_edge_idx):
        """ get edge reconstruction loss
        """
        pos_edge_prob = self.decode(g=pos_g, z=z)[pos_edge_idx] + GAE.EPSILON
        neg_edge_prob = 1.0 - self.decode(g=neg_g, z=z)[neg_edge_idx] + GAE.EPSILON
        
        pos_edge_loss = (-torch.log(pos_edge_prob)).mean()
        neg_edge_loss = (-torch.log(neg_edge_prob)).mean()
        
        return pos_edge_loss + neg_edge_loss
    
    def get_loss(self, *args, **kwargs):
        """ wrapper for loss function evaluation
        """
        return self.get_reconstruction_loss(*args, **kwargs)
```

#### VGAE

```Python
class VariationalGCNEncoder(torch.nn.Module):
    MAX_LOGSTD = 10.0
    
    def __init__(self, in_feats, out_feats, variation_scale=3):
        super().__init__()
        
        # gcn1:
        self.gcn1 = GraphConv(
            in_feats=in_feats, out_feats=2*out_feats, 
            weight=True, bias=True, 
            activation=F.relu, 
            allow_zero_in_degree=True
        )
        
        # output, mu
        self.output_mu = GraphConv(
            in_feats=2*out_feats, out_feats=out_feats, 
            weight=True, bias=True, 
            activation=None, 
            allow_zero_in_degree=True
        )
        
        # output, log(std):
        self.output_logstd = GraphConv(
            in_feats=2*out_feats, out_feats=out_feats, 
            weight=True, bias=True, 
            activation=None, 
            allow_zero_in_degree=True
        )
        
        # for sampling from encoded Gaussian
        self.output_std_scale = variation_scale
        
    def forward(self, g, x):
        """ forward propagation
        """
        h = self.gcn1(g, x)
        
        mu = self.output_mu(g, h)
        logstd = self.output_logstd(g, h)
        
        return (mu, logstd)
    
    def sample_from_encoded_gaussian(self, mu, logstd, training):
        """ sample from encoded Gaussian
        """
        if training:
            return mu + (2*torch.randn_like(logstd) - 1) * self.output_std_scale * torch.exp(logstd)
        
        return mu
    
class VGAE(GAE): 
    """变分自编码器。继承自GAE这个类，可以使用GAE里面定义的函数。
    """
    
    def __init__(self, encoder, decoder):
        super().__init__(encoder=encoder, decoder=decoder)
    
    
    def encode(self, *args, **kwargs):
        """ encode
        """
        # get encoding Gaussian:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        
        # limit standard deviation scale:
        self.__logstd__ = self.__logstd__.clamp(max=self.encoder.MAX_LOGSTD)
        
        # sample from encoding Gaussian:
        return self.encoder.sample_from_encoded_gaussian(
            self.__mu__, self.__logstd__, 
            self.training
        )

    def get_regulation_loss(self, mu=None, logstd=None):
        """ get encoding Gaussian regulation loss
        """
        mu = mu if not mu is None else self.__mu__
        logstd = logstd.clamp(max=self.encoder.MAX_LOGSTD) if not logstd is None else self.__logstd__
        
        # KL(p||q), with p as actual Gaussian and q as prior Gaussian:
        return -0.5*torch.mean(
            torch.mean(1.0 + 2*logstd - mu**2 - logstd.exp()**2, dim=1)
        )
    
    def get_loss(self, *args, **kwargs):
        """ wrapper for loss function evaluation
        """
        # TODO: the introduction of Gaussian prior regulation seems to hurt the performance
        return super().get_loss(*args, **kwargs) # + self.get_regulation_loss()
```

---

### Training, Testing and Evaluation

#### Training

```Python
def train(model, g, lr=0.01, weight_decay=5e-4, epochs=1000, validation_step_size=125):
    """ train (variational) autoencoder
    """
    # parse dataset:
    (pos, neg) = g
    (pos_g, pos_train_edge_idx, pos_valid_edge_idx) = pos
    (neg_g, neg_train_edge_idx, neg_valid_edge_idx) = neg
    
    # init optimizer:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        """Training..."""
    )
    
    # optimize:
    for i in range(epochs + 1):
        model.train()
        
        optimizer.zero_grad()
        # get encoding:
        z = model.encode(g=pos_g, x=pos_g.ndata['feat'])
        # get loss:
        train_loss = model.get_loss(z, pos_g, neg_g, pos_train_edge_idx, neg_train_edge_idx)
        
        # back propagation
        train_loss.backward()
        optimizer.step()
        
        # do validation
        if i % validation_step_size == 0:
            valid_loss = model.get_loss(z, pos_g, neg_g, pos_valid_edge_idx, neg_valid_edge_idx)
            print(
                """\tEpoch {}:\n"""
                """\t\t training / validation losses: {:.4f} / {:.4f}""".format(
                    i, 
                    train_loss.item(), valid_loss.item()
                )
            )
```

#### Testing

```Python
@torch.no_grad()
def test(model, pos_g, neg_g, pos_edge_idx, neg_edge_idx):
    model.eval()
    
    x = pos_g.ndata['feat']
    
    pos_z = model.encode(g=pos_g, x=x)
    neg_z = model.encode(g=neg_g, x=x)
    
    pos_y = pos_z.new_ones(pos_edge_idx.size)
    neg_y = neg_z.new_zeros(neg_edge_idx.size)
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(pos_g, pos_z)[pos_edge_idx]
    neg_pred = model.decoder(neg_g, neg_z)[neg_edge_idx]
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)
```

#### Evaluation

```Python
def evaluate_node_classification(
    embeddings, labels, 
    train_mask, test_mask, 
    normalize_embedding=True, 
    max_iter=1000
):
    """ use single-layer MLP for node label prediction using (variational) graph auto-encoder embeddings
    """
    # normalize:
    X = embeddings
    if normalize_embedding:
        X = normalize(embeddings)
    
    # split train-test sets:
    X_train, y_train = X[train_mask, :], labels[train_mask]
    X_test, y_test = X[test_mask, :], labels[test_mask]
    
    # build classifier:
    clf = MLPClassifier(
        random_state=42,
        hidden_layer_sizes=[32],
        max_iter=max_iter
    ).fit(X_train, y_train)
    
    # make prediction:
    preds = clf.predict(X_test)
    
    # get classification report:
    print(
        classification_report(
            y_true=y_test, y_pred=preds
        )
    )
    # get accuracy score:
    test_acc = accuracy_score(y_true=y_test, y_pred=preds)
    
    return preds, test_acc
```

---

### Results

#### GAE

```bash
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       130
           1       0.15      0.95      0.25        91
           2       0.00      0.00      0.00       144
           3       0.00      0.00      0.00       319
           4       0.34      0.92      0.49       149
           5       1.00      0.01      0.02       103
           6       0.00      0.00      0.00        64

    accuracy                           0.22      1000
   macro avg       0.21      0.27      0.11      1000
weighted avg       0.17      0.22      0.10      1000
```

#### VGAE

```bash
              precision    recall  f1-score   support

           0       0.41      0.51      0.46       130
           1       0.38      0.69      0.49        91
           2       0.78      0.88      0.83       144
           3       0.74      0.36      0.49       319
           4       0.70      0.54      0.61       149
           5       0.46      0.70      0.55       103
           6       0.28      0.36      0.32        64

    accuracy                           0.55      1000
   macro avg       0.54      0.58      0.53      1000
weighted avg       0.61      0.55      0.55      1000
```

---