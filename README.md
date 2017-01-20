###移植yolo2到caffe框架过程中用到的一些关键文件

 1. [yolo.prototxt](https://github.com/hustzxd/yolo2_to_caffe_tool/blob/master/yolo.prototxt)  
yolo2的网络配置文件，对应着darknet中的[yolo.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)编写
 2. [conver_weights_to_caffemodel.py](https://github.com/hustzxd/yolo2_to_caffe_tool/blob/master/convert_weights_to_caffemodel.py)  
将yolo2的weights文件转化为caffemodel文件  
`python conver_weights_to_caffemodel.py yolo.prototxt yolo.weight yolo.caffemodel`
 3. [compare_output.py](https://github.com/hustzxd/yolo2_to_caffe_tool/blob/master/compare_output.py)  
比较darknet和caffe各层输出的脚本文件</br>
`python compare_output.py layer_output_in_caffe layer_output_in_darknet`
 4. [save_sized_image.py](https://github.com/hustzxd/yolo2_to_caffe_tool/blob/master/save_sized_image.py)</br>
 在caffe框架中保存resize后的图片到sized_image.npy 仍然有一个问题，caffe保存的sized_image.npg 和 darknet中保存的sized_image 两个进行数据比对时并不完全一致，我使用compare_output.py 比对两个数据时，发现在一定误差范围内，数据的相似性（暂且叫这个）大概在90%多，这样应该不会影响到后边图像的特征提取
