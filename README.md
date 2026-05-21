English| [简体中文](./README_cn.md)

Getting Started with Hobot Dnn
=======

The repository contains four parts: dnn_node, dnn_node_sample, dnn_node_example, and dnn_benchmark_example. Users can implement model inference on the RDK X3 development board using the BPU processor, as well as evaluate model performance.

# dnn_node

On-board algorithm inference framework. Utilizing the BPU processor on the RDK X3 development board for AI inference, based on the lidnn algorithm inference framework and ROS2 Node for secondary development, providing a simpler and more user-friendly model integration development interface for robot applications.

# dnn_node_sample

Sample usage of dnn_node, users can **reference** it to deploy their own algorithm models.

# dnn_node_example

Example of post-processing with built-in algorithms in dnn_node.

# dnn_benchmark_example

Algorithm performance evaluation tool based on dnn_node.