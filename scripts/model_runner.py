import sys
import os
import argparse
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moral_lens.dilemma import DilemmaRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the DilemmaRunner with specified parameters.')
    parser.add_argument('--model_id', type=str, required=True, help='ID of the decision model')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory for storing results')
    parser.add_argument('--decision_run_name', type=str, default=None, help='Name for the decision run (default: None)')
    parser.add_argument('--temperature', type=float, default=None, help='Override decision temperature (default: None)')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for the sampler (default: 1)')
    parser.add_argument('--choices_filename', type=str, default=None, help='Filename for choices (default: None)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the runner (default: 1)')
    parser.add_argument('--prompts_template', type=str, default="no_reasoning", help='Template for the prompts (default: no_reasoning)')

    args = parser.parse_args()

    # Common runner configuration
    runner_kwargs = {
        "model_id": args.model_id,
        "results_dir": args.results_dir,
        "batch_size": args.batch_size,
        "prompts_template": args.prompts_template,
    }

    temperature = getattr(args, "temperature", None)
    choices_filename = getattr(args, "choices_filename", None)

    if temperature is not None:
        runner_kwargs["override_decision_temperature"] = temperature
    if choices_filename is not None:
        runner_kwargs["choices_filename"] = choices_filename

    if args.decision_run_name is None:
        args.decision_run_name = args.prompts_template

    if args.num_runs > 1 and temperature is not None and temperature > 0:
        for i in range(1, args.num_runs + 1):
            runner_kwargs["decision_run_name"] = f"{args.decision_run_name}_{i}"
            runner = DilemmaRunner(**runner_kwargs)
            asyncio.run(runner.run())
    else:
        if args.num_runs > 1:
            print("Warning: num_runs > 1 but temperature is not set or is 0. Running a single run.")
        runner_kwargs["decision_run_name"] = args.decision_run_name
        runner = DilemmaRunner(**runner_kwargs)
        asyncio.run(runner.run())
