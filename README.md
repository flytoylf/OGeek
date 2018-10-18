lgb
  1. 代码入口：sh base_run.sh。
  2. 写了很多冗余特征，跑起来非常慢，需要做一下特征选择。
  3. 线上提交后0.7154。
  4. 代码中用到了分词和词向量，需要将切词的代码换成自己的切词代码，需要自己提供词向量。
  5. 我的预训练向量是之前使用百科文本作为预料利用FastText训练出来的。
  
rnn
  1. 数据准备：python3 data/data_preprocess.py.
  2. 代码入口：python3 src/run_model.py train/pred。
  3. 写了两个网络cnn_network_core.py 和 rnn_network_core.py，使用的时候把相应文件改为network_core.py
  4. 之前使用纯cnn提交了一版，线上成绩0.7059。gru + cnn + attention跑的太慢，线上还未提交。
  5. 代码需要提供预训练向量，如果没有预训练向量需要将is_use_pretrained_word_embedding置为False。
  6. 网络结构在network_core.py中，这里可以设计自己的网络结构进行优化。
  
  
由于没有太多时间搞，代码很多地方写的比较糙，这里提供一个思路，欢迎交流！
