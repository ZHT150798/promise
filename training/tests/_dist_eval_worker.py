"""CPU/gloo distributed evaluator aggregation check.

This worker is launched by test_evaluator_distributed.py through
`accelerate launch --cpu --num_processes N`.

This worker calls the production Evaluator aggregation path with FakeValDataset
so the distributed test covers the same gather/barrier/broadcast helpers used
by validation.
"""

import argparse
import json
import os

from accelerate import Accelerator
from promise_train.trainer.evaluator import Evaluator
from torch.utils.data import DataLoader, Dataset


class FakeValDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, float]:
        return {"id": idx, "metric": float(idx)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--skip_fid", type=int, default=1)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    accelerator = Accelerator()
    dataloader = DataLoader(FakeValDataset(args.n_samples), batch_size=1, shuffle=False)
    evaluator = Evaluator(
        accelerator=accelerator,
        output_dir=args.out_dir,
        skip_fid=bool(args.skip_fid),
    )
    output = evaluator.evaluate_scalar_loader(
        dataloader,
        prepare_loader=True,
        run_fid_barriers=not bool(args.skip_fid),
    )
    with open(os.path.join(args.out_dir, f"rank_{accelerator.process_index}.json"), "w") as file:
        json.dump(output, file)


if __name__ == "__main__":
    main()
