import unittest

import torch
from torch import nn

from mutimodal.transfromer.MainNet import MainNet
from mutimodal.transfromer.Conv1D import MyConv1D
from PositionalEncoding import PositionalEncoding
from mutimodal.transfromer.TotalNet import TotalNet


class TestTotalNet(unittest.TestCase):
    def setUp(self):
        """创建模型实例和测试数据"""
        self.model = TotalNet()
        self.batch_size = 4
        self.time_steps = 32
        self.original_dim = 256

        # 创建模拟输入数据
        self.d_eeg = torch.randn(self.batch_size, self.time_steps, self.original_dim)
        self.d_face = torch.randn(self.batch_size, self.time_steps, self.original_dim)
        self.d_ecg = torch.randn(self.batch_size, self.time_steps, self.original_dim)

    def test_model_initialization(self):
        """测试模型是否能正确初始化"""
        self.assertIsInstance(self.model, TotalNet)
        self.assertIsInstance(self.model.conv1_e, MyConv1D)
        self.assertIsInstance(self.model.conv1_f, MyConv1D)
        self.assertIsInstance(self.model.conv1_c, MyConv1D)
        self.assertIsInstance(self.model.pos, PositionalEncoding)
        self.assertIsInstance(self.model.main_e, MainNet)
        self.assertIsInstance(self.model.main_f, MainNet)
        self.assertIsInstance(self.model.main_c, MainNet)
        self.assertIsInstance(self.model.f, nn.Sequential)

    def test_forward_pass(self):
        """测试前向传播是否正常运行"""
        try:
            output = self.model(self.d_eeg, self.d_face, self.d_ecg)
            self.assertIsNotNone(output)
        except Exception as e:
            self.fail(f"前向传播失败: {e}")

    def test_output_shape(self):
        """测试输出形状是否正确"""
        output = self.model(self.d_eeg, self.d_face, self.d_ecg)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_output_range(self):
        """测试输出值是否在0-1范围内"""
        output = self.model(self.d_eeg, self.d_face, self.d_ecg)
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())

    def test_different_input_shapes(self):
        """测试不同序列长度的输入"""
        # 创建不同序列长度的输入
        d_eeg = torch.randn(self.batch_size, 40, self.original_dim)
        d_face = torch.randn(self.batch_size, 30, self.original_dim)
        d_ecg = torch.randn(self.batch_size, 50, self.original_dim)

        try:
            output = self.model(d_eeg, d_face, d_ecg)
            self.assertEqual(output.shape, (self.batch_size, 1))
        except Exception as e:
            self.fail(f"处理不同序列长度失败: {e}")

    def test_model_parameters(self):
        """测试模型是否有可训练参数"""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0, "模型没有可训练参数")

        # 检查每个参数都有梯度
        for param in params:
            self.assertTrue(param.requires_grad, "参数未设置requires_grad=True")

    def test_model_on_gpu(self):
        """测试模型在GPU上的运行情况"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_gpu = TotalNet().to(device)
            d_eeg_gpu = self.d_eeg.to(device)
            d_face_gpu = self.d_face.to(device)
            d_ecg_gpu = self.d_ecg.to(device)

            try:
                output = model_gpu(d_eeg_gpu, d_face_gpu, d_ecg_gpu)
                self.assertEqual(output.shape, (self.batch_size, 1))
            except Exception as e:
                self.fail(f"GPU前向传播失败: {e}")
        else:
            self.skipTest("没有可用的GPU设备")

    def test_model_with_gradients(self):
        """测试模型是否能计算梯度"""
        # 创建需要梯度的输入
        d_eeg = self.d_eeg.clone().requires_grad_(True)
        d_face = self.d_face.clone().requires_grad_(True)
        d_ecg = self.d_ecg.clone().requires_grad_(True)

        output = self.model(d_eeg, d_face, d_ecg)

        # 计算梯度
        try:
            loss = output.sum()
            loss.backward()

            # 检查输入梯度
            self.assertIsNotNone(d_eeg.grad)
            self.assertIsNotNone(d_face.grad)
            self.assertIsNotNone(d_ecg.grad)

            # 检查模型参数梯度
            for name, param in self.model.named_parameters():
                self.assertIsNotNone(param.grad, f"参数 {name} 没有梯度")
        except Exception as e:
            self.fail(f"梯度计算失败: {e}")


if __name__ == "__main__":
    unittest.main()
