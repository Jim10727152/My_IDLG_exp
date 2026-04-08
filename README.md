# My_IDLG_exp

IDLG 為 gradient leakage attack 的其中一種，藉由梯度使用迭代的方式還原原始的輸入
我嘗試使用差分隱私去做防禦
  IDLG_MNIST_Defense.py  為對所有梯度加入雜訊
  IDLG_MNIST_Defense_bias.py  為僅對buas梯度加入雜訊
  IDLG_MNIST_Defense_first_layer.py  為對第一層的參數加入雜訊

