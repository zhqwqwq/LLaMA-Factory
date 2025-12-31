set -eui pipefail


MODEL_PATHS=(
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_20"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_60"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_100"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_140"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_180"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_220"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_260"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/Llama3_GRPO_run2/actor/global_step_300"

    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-480"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-960"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-1440"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-1920"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-2400"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-2880"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-3360"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-3840"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-4320"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_300_47rollout/checkpoint-4800"
)

for model_path in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model at: $model_path"
    save_dir="${model_path}/mmlu_eval"
    llamafactory-cli eval /cephfs/zhanghuaqing/RL/LLaMA-Factory/experiments/10.27_gsm8k_forgetting_MMLU/eval.yaml \
        model_name_or_path="$model_path" \
        save_dir="$save_dir"
done