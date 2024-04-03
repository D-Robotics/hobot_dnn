# Changelog for package dnn_benchmark_example

tros_2.3.0 (2024-03-27)
------------------
1. 新增适配ros2 humble零拷贝。
2. 新增中英双语Readme。
3. 适配重构dnn_node。
4. 零拷贝通信使用的qos的Reliability由RMW_QOS_POLICY_RELIABILITY_RELIABLE（rclcpp::QoS()）变更为RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT（rclcpp::SensorDataQoS()）。

tros_2.2.2 (2023-12-22)
------------------

tros_2.0.1 (2023-07-14)
------------------
1. 规范Rdkultra产品名。

tros_2.0.0rc1 (2023-05-23)
------------------
1. 修复图片回灌出错问题


tros_2.0.0 (2023-05-11)
------------------
1. 更新package.xml，支持应用独立打包
2. 更新应用启动launch脚本


tros_1.1.6b (2023-3-03)
------------------
1. 修复readme文档错误导致使用错误问题。


tros_1.1.6a (2023-2-16)
------------------
1. 支持x86版本部署，适配x86版本的dnn node，新增x86版本编译选项。


tros_1.1.2 (2022-9-28)
------------------
1. 新增dnn_node_sample package，为dnn_node的使用示例，用户可以参考部署自己的算法模型。
