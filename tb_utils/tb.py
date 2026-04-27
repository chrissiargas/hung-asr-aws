from tensorboard import program
import os
import socket
from config.parser import Parser
from tb_utils.cleanup import cleanup_short_runs
from datetime import datetime
import tempfile

conf = Parser()
conf.get_args()

task = 'dual_fusion_checkpoints'
checkpoint_dir = os.path.join(os.path.expanduser('~'),
                              conf.dual_fuse_args.checkpoint_path,
                              task)

cutoff_date = datetime(2026, 1, 5, 0, 0)

runs = cleanup_short_runs(checkpoint_dir, 0, cutoff_date, dry_run=True, keep_after=True)
print(runs)

if __name__ == "__main__":
    symlink_dir = tempfile.mkdtemp(prefix="tb_clean_runs_")

    for i, run_path in enumerate(runs):
        symlink_name = '|'.join(run_path.split('/')[-3:])
        symlink_path = os.path.join(symlink_dir, symlink_name)
        try:
            os.symlink(run_path, symlink_path)
        except FileExistsError:
            pass

    print(f"Created clean symlink tree at: {symlink_dir}")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', symlink_dir])
    tb.main()