# Set the visible devices to 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Paired setting
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no \
    inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --data_dir "zalando-hd-resized" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0

# Unpaired setting
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no \
    inference.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" --unpaired --data_dir "zalando-hd-resized" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0



# #DressCode
# ##upper_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "upper_body"

# ##lower_body
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "lower_body"

# ##dresses
# accelerate launch inference_dc.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "result" --unpaired --data_dir "/home/omnious/workspace/yisol/DressCode" \
#     --seed 42 --test_batch_size 2 --guidance_scale 2.0 --category "dresses"
