# LLM
## deepseek-VL2-tiny-fake_news中
运行时下载DeepSeek-VL2与deepseek-VL2-tiny模型
运行时遇到dtype问题，将forward() 函数修改一下。
DeepSeek-VL2/deepseek-vl2/models/modeling_deepseek.py的911行， y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)
