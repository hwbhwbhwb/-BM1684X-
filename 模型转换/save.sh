model_transform.py \
    --model_name Qwen3_Embedding_0.6B \
    --model_def ../onnx/model.onnx \
    --input_shapes \
        "input_ids=[1,1024]" \
        "attention_mask=[1,1536]" \
        "position_ids=[1,1024]" \
        "past_key_values.*.key=[1,8,512,128]" \
        "past_key_values.*.value=[1,8,512,128]" \
    --keep_aspect_ratio \
    --mlir Qwen3_Embedding_0.6B_1024.mlir

model_transform.py \
    --model_name Qwen3_Embedding_0.6B \
    --model_def ../onnx_output/model.onnx \
    --input_shapes "[[1,1024], [1,1536], [1,1024], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128], [1,8,512,128]]" \
    --keep_aspect_ratio \
    --mlir Qwen3_Embedding_0.6B_1024.mlir

model_transform.py \
    --model_name Qwen3_Embedding_0.6B \
    --model_def ../onnx_output/model.onnx \
    --input_shapes "[[1,128], [1,128]]" \
    --keep_aspect_ratio \
    --mlir Qwen3_Embedding_0.6B_1024.mlir

model_deploy.py \
    --mlir Qwen3_Embedding_0.6B_1024.mlir \
    --quantize F16 \
    --processor bm1684x \
    --model Qwen3_Embedding_0.6B_1684x_1024_f16.bmodel

model_deploy.py \
    --mlir Qwen3_Embedding_0.6B_1024.mlir \
    --quantize F32 \
    --processor bm1684x \
    --model Qwen3_Embedding_0.6B_1684x_1024_f32.bmodel

tpuc-opt Qwen3_Embedding_0.6B_bm1684x_f32_final.mlir \
    --op-divide \
    --codegen="model_file=Qwen3_Embedding_0.6B_1684x_1024_f32.bmodel embed_debug_info=False model_version=latest bmodel_only=False gdma_check=True" -o /dev/null
