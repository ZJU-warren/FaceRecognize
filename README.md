### 模型测试
进入./Code/C_Recognize文件夹
输入 python3 Evaluate.py
将自动生成在LFW数据上测试的结果和ROC、CMC图

### 模型应用
在./DataSet/App下添加新的文件夹，以人名进行命名，里面放入一张或多张该人的照片
进入./Code/D_Application文件夹
输入 python3 VideoCapture.py
检测到人脸后，会暂停画面框出，并给出已有人脸中与其最接近的一张
暂停后按空格继续检测
过程中，按q可以退出程序
