python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/alora/20task.json
python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/llora/20task.json

python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/alora/10task.json
python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/llora/10task.json

python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/alora/50task.json
python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/llora/50task.json

python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/alora/5task.json
python main.py --device 3 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/llora/5task.json


# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/5task.json
# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/10task.json
# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/20task.json
# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/50task.json

# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/5task.json
# python main.py --device 2 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/vanilla/5task.json

# python main.py --device 1 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/rank1/20task.json
# python main.py --device 1 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/rank5/20task.json
# python main.py --device 1 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/rank10/20task.json
# python main.py --device 1 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/rank20/20task.json
# python main.py --device 1 --config /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/rank64/20task.json

# for config in /data/wfq/Lora/RLora_v3/configs/imagetnet_a/dlora/layer/*.json
# do
#     echo "Running $config"
#     python main.py --device 3 --config $config
# done