import onnx

model_path = "/deltadisk/guestnju/tpu/Qwen3-Embedding-0.6B-ONNX/onnx/model.onnx"
model = onnx.load(model_path)
op_types = set(node.op_type for node in model.graph.node)
print("模型使用的算子：")
for op in sorted(op_types):
    print(" -", op)
    
# (less) guestnju@kris-3090x2:~/tpu$ python list_onnx_op.py (hg版本，其他人转的lower版本，似乎也没有onnx-community专业，没有KV input等缓存机制)
# 模型使用的算子：
#  - Add
#  - Cast
#  - Concat
#  - Constant
#  - ConstantOfShape
#  - Cos
#  - Div
#  - Equal
#  - Expand
#  - Gather
#  - Greater
#  - MatMul
#  - Mul
#  - Neg
#  - Pow
#  - Range
#  - ReduceMean
#  - Reshape
#  - ScatterND
#  - Shape
#  - Sigmoid
#  - Sin
#  - Slice
#  - Softmax
#  - Sqrt
#  - Transpose
#  - Unsqueeze
#  - Where

# (less) guestnju@kris-3090x2:~/tpu$ python list_onnx_op.py （my版本，自己用export_model_onnx.py导出的onnx文件算子）
# 模型使用的算子：
#  - Add
#  - ArgMax
#  - Cast
#  - Clip
#  - Concat
#  - Constant
#  - ConstantOfShape
#  - Cos
#  - Div
#  - Equal
#  - Expand
#  - Gather
#  - GatherElements
#  - Greater
#  - MatMul
#  - Mul
#  - Neg
#  - Pow
#  - Range
#  - ReduceL2
#  - ReduceMax
#  - ReduceMean
#  - Reshape
#  - ScatterND
#  - Shape
#  - Sigmoid
#  - Sin
#  - Slice
#  - Softmax
#  - Sqrt
#  - Squeeze
#  - Sub
#  - Tile
#  - Transpose
#  - Unsqueeze
#  - Where

# (less) guestnju@kris-3090x2:~/tpu$ python list_onnx_op.py （hg版本，hg上onnx-community上下的模型）
# 模型使用的算子：
#  - Add
#  - Cast
#  - Concat
#  - Constant
#  - ConstantOfShape
#  - Equal
#  - Expand
#  - Gather
#  - Less
#  - MatMul
#  - Mul
#  - MultiHeadAttention
#  - Range
#  - Reshape
#  - RotaryEmbedding
#  - Shape
#  - Sigmoid
#  - SimplifiedLayerNormalization
#  - SkipSimplifiedLayerNormalization
#  - Slice
#  - Squeeze
#  - Sub
#  - Tile
#  - Transpose
#  - Unsqueeze
#  - Where