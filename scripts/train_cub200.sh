# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/alora/5task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/llora/5task.json

python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/alora/20task.json
python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/llora/20task.json

# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/alora/10task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/llora/10task.json


# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/alora/50task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/llora/50task.json


# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/vanilla/5task.json


# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/dlora/rank1/20task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/dlora/rank5/20task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/dlora/rank10/20task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/dlora/rank20/20task.json
# python main.py --device 0 --config /data/wfq/Lora/RLora_v3/configs/cub200/dlora/rank64/20task.json



# for config in /data/wfq/Lora/RLora_v3/configs/cub200/dlora/layer/*.json
# do
#     echo "Running $config"
#     python main.py --device 2 --config $config
# done