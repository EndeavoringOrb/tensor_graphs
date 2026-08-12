# main.py
import argparse
import os

import tensor_graphs
from safetensors.torch import load_file

from train import AdvancedAgent, AgentDelegate


def main():
    parser = argparse.ArgumentParser(description="Run AutoRegressive LLM Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-270m",
        help="gemma-3-270m, qwen-3.6-35b-a3b, deepseek-v4",
    )
    parser.add_argument("--model-path", type=str, default="models/google/gemma-3-270m")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to runs/N to load the agent from",
    )
    parser.add_argument(
        "--tokens", type=int, default=20, help="Number of tokens to generate"
    )
    args = parser.parse_args()

    agent = AdvancedAgent(hidden_dim=64)
    if args.run_dir:
        model_file = os.path.join(args.run_dir, "model.safetensors")
        if os.path.exists(model_file):
            agent.load_state_dict(load_file(model_file))
            print(f"Loaded trained delegate agent from {model_file}")

    agent.eval()
    delegate = AgentDelegate(agent)

    print(f"Loading {args.model} via LLMSession...")
    session = tensor_graphs.LLMSession(args.model, args.model_path, delegate)

    # Default starting token (BOS)
    tokens = [2]

    print("Generating...")
    for step in range(args.tokens):
        next_token = session.generate_step(tokens)
        if next_token == -1:
            print("Max sequence length reached.")
            break
        tokens.append(next_token)
        print(f"Step {step + 1} | Token: {next_token}")


if __name__ == "__main__":
    main()
