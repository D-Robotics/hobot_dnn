# Changelog for package dnn_node_example

tros_2.4.0 (2024-07-15)
------------------
1. 新增支持x5平台支持。
2. 新增x5工具链参考算法适配。

tros_2.3.0rc1 (2024-04-12)
------------------
1. 修复未设置`CAM_TYPE`情况下example启动失败的问题。


tros_2.3.0 (2024-03-27)
------------------
1. 新增适配ros2 humble零拷贝。
2. 新增中英双语Readme。
3. 适配重构dnn_node。
4. 新增分割任务发布ai_msgs，支持在Web中显示渲染效果。
5. 零拷贝通信使用的qos的Reliability由RMW_QOS_POLICY_RELIABILITY_RELIABLE（rclcpp::QoS()）变更为RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT（rclcpp::SensorDataQoS()）。

tros_2.2.2 (2023-12-22)
------------------
1. 更新CMakeLists.txt，解决arm平台打包失败问题。

tros_2.2.1 (2023-07-14)
------------------
1. 规范Rdkultra产品名。

tros_2.2.0 (2023-07-07)
------------------
1. 适配RDK Ultra平台。

tros_2.1.0 (2023-06-20)
------------------
1. 删除针对于多模型的model_name配置项，从模型文件中自动获取model_name。


tros_2.0.0 (2023-05-11)
------------------
1. 更新package.xml，支持应用独立打包
2. 更新应用启动launch脚本


tros_1.1.6 (2023-3-15)
------------------
1. 更新示例使用的路径，统一个平台使用/opt路径下的模型路径。
2. 修复dnn_node_example没有默认配置的问题。


tros_1.1.6b (2023-3-03)
------------------
1. 修复readme文档错误导致使用错误问题。


tros_1.1.6a (2023-2-16)
------------------
1. 适配x86版本的dnn node，新增x86版本编译选项。
2. 新增launch文件使用fb回灌模式选项。


tros_1.1.4 (2022-12-27)
------------------
1. 删除未使用的GetNV12Tensor接口。


tros_1.1.3 (2022-11-09)
------------------
1. 删除接口GetNV12PyramidFromNV12Mat，统一使用dnn_node内部GetNV12PyramidFromNV12Img接口获取NV12PyramidInput数据。
2. 前处理中统一将图片padding到左上方。


tros_1.1.2rc2 (2022-10-26)
------------------
1. 修复回灌启动脚本中默认算法无效的问题。


tros_1.1.2rc1 (2022-10-17)
------------------
1. 算法后处理中使用dnn node重构后的算法输出解析方法。
2. 删除cv_bridge编译宏控制，默认使用cv_bridge。
3. 删除同步或异步推理模式配置项is_sync_mode，使用效率更高的异步推理模式。
4. 更新yolov2、yolov3、yolov5、fcos算法参数可外部配置方式。
5. 支持用户通过配置launch文件中"msg_pub_topic_name"成员，指定发布的AI消息topic。
6. 使用板端安装在/app/model/basic/路径下的模型文件。


hhp_1.0.6 (2022-08-31)
------------------
1. example中算法前处理使用hobot_cv进行图片resize。

hhp_1.0.5 (2022-08-11)
------------------
1. 修复发布的ROS Msg中算法推理输出帧率数据错误的问题。
2. 运行异常时，使用rclcpp::shutdown()触发退出后终止当前状态，避免扩大异常范围。


hhp_1.0.4 (2022-07-26)
------------------
1. 适配hhp_1.0.4版本dnn_node。


v1.0.1 (2022-06-23)
------------------
1. 发布的AI msg中增加perf信息输出，用于推理性能分析。
