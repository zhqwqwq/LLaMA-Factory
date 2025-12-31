set -eui pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATHS=(
        # "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_20"
    # "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_40"
    # "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_60"
    # "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_80"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_dpsk_16rollouts/checkpoint-1280"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-160"
)

MODEL_PATHS=(
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-320"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-480"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-640"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-800"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-960"
    "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1120"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_20"
    # "/cephfs/zhanghuaqing/RL/LLaMA-Factory/saves/Llama3B_gsm8k_SFT_rltrained_16rollout/checkpoint-1120"
    "/cephfs/zhanghuaqing/RL/rllm_deepscaler/checkpoints/deepscaler/deepscaler-1.5b-8k_run2/actor/global_step_40"

)
for model_path in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model at: $model_path"
    save_dir="${model_path}/mmlu_eval"
    llamafactory-cli eval /cephfs/zhanghuaqing/RL/LLaMA-Factory/experiments/10.27_gsm8k_forgetting_MMLU/eval.yaml \
        model_name_or_path="$model_path" \
        save_dir="$save_dir"
done