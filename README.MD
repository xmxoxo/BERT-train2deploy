# BERT模型从训练到部署全流程

Tag: BERT 训练 部署
## 缘起
在群里看到许多朋友在使用BERT模型，网上多数文章只提到了模型的训练方法，后面的生产部署及调用并没有说明。
这段时间使用BERT模型完成了从数据准备到生产部署的全流程，在这里整理出来，方便大家参考。

在下面我将以一个“手机评论的情感分类”为例子，简要说明从训练到部署的全部流程。最终完成后可以使用一个网页进行交互，实时地对输入的评论语句进行分类判断。

## 基本架构
基本架构为：
```mermaid
graph LR
A(BERT模型服务端) --> B(API服务端)
B-->A
B --> C(应用端)
C-->B
```
架构说明：
**BERT模型服务端**
	加载模型，进行实时预测的服务；
    使用的是 BERT-BiLSTM-CRF-NER 

**API服务端** 
	调用实时预测服务，为应用提供API接口的服务，用flask编写； 

**应用端**
	最终的应用端；
	我这里使用一个HTML网页来实现；

项目完整源码地址：[BERT从训练到部署git源码](https://github.com/xmxoxo/BERT-train2deploy)

## 关键节点
主要包括以下关键节点：
* 数据准备
* 模型训练
* 模型格式转化
* 服务端部署与启动
* API服务编写与部署
* 客户端(网页端的编写与部署）


## 数据准备
这里用的数据是手机的评论，数据比较简单,三个分类: -1,0,1 表示负面，中性与正面情感
数据格式如下：

```
1	手机很好，漂亮时尚，赠品一般
1	手机很好。包装也很完美，赠品也是收到货后马上就发货了
1	第一次在第三方买的手机 开始很担心 不过查一下是正品 很满意
1	很不错 续航好 系统流畅
1	不知道真假，相信店家吧
1	快递挺快的，荣耀10手感还是不错的，玩了会王者还不错，就是前后玻璃，
1	流很快，手机到手感觉很酷，白色适合女士，很惊艳！常好，运行速度快，流畅！
1	用了一天才来评价，都还可以，很满意
1	幻影蓝很好看啊，炫彩系列时尚时尚最时尚，速度快，配送运行?做活动优惠买的，开心?
1	快递速度快，很赞！软件更新到最新版。安装上软胶保护套拿手上不容易滑落。
0	手机出厂贴膜好薄啊，感觉像塑料膜。其他不能发表
0	用了一段时间，除了手机续航其它还不错。
0	做工一般
1	挺好的，赞一个，手机很好，很喜欢
0	手机还行，但是手机刚开箱时屏幕和背面有很多指纹痕迹，手机壳跟**在地上磨过似的，好几条印子。要不是看在能把这些痕迹擦掉，和闲退货麻烦，就给退了。就不能规规矩矩做生意么。还有送的都是什么吊东西，运动手环垃圾一比，贴在手机后面的固定手环还**是塑料的渡了一层银色，耳机也和图片描述不符，碎屏险已经注册，不知道怎么样。讲真的，要不就别送或者少送，要不，就规规矩矩的，不然到最后还让人觉得不舒服。其他没什么。
-1	手机整体还可以，拍照也很清楚，也很流畅支持华为。给一星是因为有缺陷，送的耳机是坏的！评论区好评太多，需要一些差评来提醒下，以后更加注意细节，提升质量。
0	前天刚买的，  看着还行， 指纹解锁反应不错。
1	高端大气上档次。
-1	各位小主，注意啦，耳机是没有的，需要单独买
0	外观不错，感觉很耗电啊，在使用段时间评价
1	手机非常好，很好用
-1	没有发票，图片与实物不一致
1	习惯在京东采购物品，方便快捷，及时开票进行报销，配送员服务也很周到！就是手机收到时没有电，感觉不大正常
1	高端大气上档次啊！看电影玩游戏估计很爽！屏幕够大！

```

数据总共8097条，按6:2:2的比例拆分成train.tsv,test.tsv ,dev.tsv三个数据文件

## 模型训练
训练模型就直接使用BERT的分类方法，把原来的`run_classifier.py` 复制出来并修改为 `run_mobile.py`。关于训练的代码网上很多，就不展开说明了，主要有以下方法：

```python
#-----------------------------------------
#手机评论情感分类数据处理 2019/3/12 
#labels: -1负面 0中性 1正面
class SetimentProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""

    """
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label_list.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as fd:
            pickle.dump(self.labels, fd)
    """
    return ["-1", "0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0: 
        continue
      guid = "%s-%s" % (set_type, i)

      #debug (by xmxoxo)
      #print("read line: No.%d" % i)

      text_a = tokenization.convert_to_unicode(line[1])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples
#-----------------------------------------
```
然后添加一个方法：

```python
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "setiment": SetimentProcessor, #2019/3/27 add by Echo
  }
```

**特别说明**，这里有一点要注意，在后期部署的时候，需要一个label2id的字典，所以要在训练的时候就保存起来，在`convert_single_example`这个方法里增加一段：

```python
  #--- save label2id.pkl ---
  #在这里输出label2id.pkl , add by xmxoxo 2019/2/27
  output_label2id_file = os.path.join(FLAGS.output_dir, "label2id.pkl")
  if not os.path.exists(output_label2id_file):
    with open(output_label2id_file,'wb') as w:
      pickle.dump(label_map,w)

  #--- Add end ---
```

这样训练后就会生成这个文件了。

使用以下命令训练模型，目录参数请根据各自的情况修改：

```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

sudo python run_mobile.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```
由于数据比较小，训练是比较快的，训练完成后，可以在输出目录得到模型文件，这里的模型文件格式是.ckpt的。
训练结果：
```
eval_accuracy = 0.861643
eval_f1 = 0.9536328
eval_loss = 0.56324786
eval_precision = 0.9491279
eval_recall = 0.9581805
global_step = 759
loss = 0.5615213

```

可以使用以下语句来进行预测：

```shell
sudo python run_mobile.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME

```


## 模型格式转化

由于模型服务端使用的是.pb格式的模型文件，所以需要把生成的ckpt格式模型文件转换成.pb格式的模型文件。
我这里提供了一个转换工具:`freeze_graph.py`，使用如下：

```shell
usage: freeze_graph.py [-h] -bert_model_dir BERT_MODEL_DIR -model_dir
                       MODEL_DIR [-model_pb_dir MODEL_PB_DIR]
                       [-max_seq_len MAX_SEQ_LEN] [-num_labels NUM_LABELS]
                       [-verbose]
```

这里要注意的参数是：

* `model_dir` 就是训练好的.ckpt文件所在的目录
* `max_seq_len` 要与原来一致；
* `num_labels` 是分类标签的个数,本例中是3个


```shell
python freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -max_seq_len 128 \
    -num_labels 3

```

执行成功后可以看到在`model_dir`目录会生成一个`classification_model.pb` 文件。

## 服务端部署与启动

现在可以安装服务端了，使用的是 BERT-BiLSTM-CRF-NER，项目地址：[https://github.com/macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)

下载代码并安装 ：

```shell
cd O:\开源系统\NLP\
git clone https://github.com/macanv/BERT-BiLSTM-CRF-NER
cd BERT-BiLSTM-CRF-NER
py setup.py install
```

安装完后运行服务，同时指定监听 HTTP 8091端口，并使用GPU 1来跑；

```shell
cd /mnt/sda1/transdat/bert-demo/bert/bert_svr

export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 
```
**注意**：port 和 port_out 这两个参数是API调用的端口号，
默认是5555和5556,如果你准备部署多个模型服务，那一定要指定自己的端口号，避免冲突。
我这里是改为： 5575 和 5576

如果报错没运行起来，可能是有些模块没装上,都是 bert_base/server/http.py里引用的，装上就好了：

```
sudo pip install flask 
sudo pip install flask_compress
sudo pip install flask_cors
sudo pip install flask_json
```

我这里的配置是2个GTX 1080 Ti，这个时候双卡的优势终于发挥出来了，GPU 1用于预测，GPU 0还可以继续训练模型。

运行服务后会自动生成很多临时的目录和文件，为了方便管理与启动，可建立一个工作目录，并把启动命令写成一个shell脚本。
这里创建的是`mobile_svr\bertsvr.sh` ，这样可以比较方便地设置服务器启动时自动启动服务，另外增加了每次启动时自动清除临时文件

代码如下：

```shell
#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动BERT分类模型 

echo '正在启动 BERT mobile svr...'
cd /mnt/sda1/transdat/bert-demo/bert/mobile_svr
sudo rm -rf tmp*

export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 

```

### 端口测试

可以使用curl命令来测试一下

```

curl -X POST http://192.168.15.111:8091/encode \
  -H 'content-type: application/json' \
  -d '{"id": 111,"texts": ["总的来说，这款手机性价比是特别高的。","槽糕的售后服务！！！店大欺客"], "is_tokenized": false}'

```

执行结果：
```
>   -H 'content-type: application/json' \
>   -d '{"id": 111,"texts": ["总的来说，这款手机性价比是特别高的。","槽糕的售后服务！！！店大欺客"], "is_tokenized": false}'
{"id":111,"result":[{"pred_label":["1","-1"],"score":[0.9974544644355774,0.9961422085762024]}],"status":200}

```
可以看到对应的两个评论，预测结果一个是1，另一个是-1，计算的速度还是非常很快的。
通过这种方式来调用还是不太方便，知道了这个通讯方式，我们可以用flask编写一个API服务，为所有的应用统一提供服务。


## API服务编写与部署
未完待续...

## 客户端(网页端的编写与部署）
未完待续...

