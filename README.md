#风力发电风速预测  

自己论文的源代码

##方法1，基于LSTM神经网络的风速预测  
两层lstm神经网络的时间序列预测，利用lstm，避免传递过程中的梯度消失。  
代码：lstm.py。使用keras搭建
##方法2，基于CNN和RNN融合模型+FRS+风速软测量的风速预测   
***模糊粗糙集属性约简+风机软测量方法的输入参数融合***  
模糊粗糙集属性约简修改的matlab算法，python实现。
![image](https://github.com/lab135-ncepu/-/blob/master/%E8%BE%93%E5%85%A5%E5%8F%82%E6%95%B0%E7%A1%AE%E5%AE%9A.JPG)
***Clstm神经网络模型***    
clstm神经网络模型用的pytorch搭建，pytorch的确简单好用
![image](https://github.com/lab135-ncepu/-/blob/master/%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B.JPG)
***整体预测框图***
![image](https://github.com/lab135-ncepu/-/blob/master/%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B.JPG)
***风速预测结果***
![image](https://github.com/lab135-ncepu/-/blob/master/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.JPG)
