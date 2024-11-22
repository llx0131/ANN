# #Read me

File list

工作目录：~/Documents/pytorch计算/WWTPs温度扰动课题

1）首先打开工作目录下的“1 45个特征ANN网络”目录。

- 使用pycharm运行“(1) All_ANN_train_select.py”文件，该文件会读取工作目录下的“Environment_select.csv”和“Microbiom-ASV-alpha_select.csv”文件进行ANN人工神经网络训练和保存模型。
- 模型保存路径为工作目录下的“result/alpha“，注：为每个随机种子保存了相应的模型。
- 

2）随后打开工作目录下的“1 45个特征ANN网络”目录。

- 使用pycharm运行“(3) 所有样本作测试集，TestAndParameters_select.py”，该文件仍旧会读取工作目录下的“Environment_select.csv”和“Microbiom-ASV-alpha_select.csv”文件，并选择所有的样本进行预测。
- 预测结果保存在工作目录下的“results/alpha/analysis新45目录中
- 同时将所有的预测结果保存在“results/alpha/analysis新45“目录下的“未扰动预测.csv”文件中，随后将此文件复制到“results/alpha/analysis_dis“目录下，用于未来的”未扰动-扰动做图“。
- 此时未扰动之前的模型的预测已经完成。
- 接下来的任务是，1进行扰动，2用第一步构建好的模型利用扰动后的环境数据进行预测。3做图

3～5步为一个扰动循环，要一起运行。例如扰动1度，需要3～5步跑一次。

3）环境数据ENV扰动（扰动1度、5度、10度）

- 打开工作目录下的“2 Environment元数据扰动编辑”目录。
- 在此目录下新建R，依次运行“source('1 Preprocess_dis.R')”、perturb_Tem("归一化后.csv", "原始.csv", 1, 'Environment_select_DIS.csv')。该代码会构建扰动Tem1的环境数据。并保存到该目录下的“Tem1”目录中。
- Tem5、Tem10类似。

4）随后打开工作目录下的“3 温度扰动预测代码”目录，进行扰动之后的预测。

- 运行相应的“(3) 所有样本作测试集，TestAndParameters_select_X.py”代码。
- 如1度就运行“”(3) 所有样本作测试集，TestAndParameters_select_1.py“。
- 该代码会将预测数据保存在“results/alpha/analysis_dis”目录中。

5）做图——“未扰动-扰动后”

- 依次运行“results/alpha/analysis_dis”目录下的“1 得到行名和列名.R”和“2 未扰动-扰动后R做图分析.R”代码做图即可，相应的做图结果会依次保存到对应的文件夹中。