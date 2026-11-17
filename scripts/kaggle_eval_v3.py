"""Build revision-pinned Kaggle workers for paired held-out V3 evaluation."""
import inspect
import json
import re


def prepare_eval_inputs(config, input_root, work_root):
    """Verify exact, finite final weights and preserve the original job bytes."""
    import hashlib
    import json
    import shutil
    from pathlib import Path
    import torch

    inputs, output = Path(input_root), Path(work_root)
    name = config['job_id']
    checkpoints = list(inputs.rglob(name + '.last.pt'))
    jobs = list(inputs.rglob(name + '.json'))
    if len(checkpoints) != 1 or len(jobs) != 1 or checkpoints[0].parent != jobs[0].parent:
        raise ValueError('Evaluation input is missing or ambiguous')
    checkpoint, job_file = checkpoints[0], jobs[0]
    def digest(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest(checkpoint) != config['checkpoint_sha256'] or digest(job_file) != config['job_sha256']:
        raise ValueError('Evaluation checkpoint or job hash mismatch')
    job = json.loads(job_file.read_text(encoding='utf-8'))
    if any(job.get(key) != config[key] for key in ('job_id', 'variant', 'seed', 'source_revision')):
        raise ValueError('Evaluation job identity differs from the dispatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('schema_version') != 3 or state.get('job') != job:
        raise ValueError('Checkpoint belongs to another job')
    if state.get('status') not in ('converged', 'completed') or not state.get('best_state'):
        raise ValueError('Evaluation requires final validation-selected weights')
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(v) for v in value)
        return True
    if not state.get('model_state') or not finite(state):
        raise ValueError('Checkpoint contains nonfinite tensors or no model weights')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / job_file.name
    if destination.exists() and destination.read_bytes() != job_file.read_bytes():
        raise ValueError('Refusing to overwrite a different job configuration')
    if destination.resolve() != job_file.resolve():
        shutil.copyfile(job_file, destination)
    result = {'job_id': name, 'checkpoint': str(checkpoint),
              'job_file': str(destination), 'checkpoint_sha256': config['checkpoint_sha256'],
              'job_sha256': config['job_sha256'], 'source_kernel': config['source_kernel'],
              'status': state['status'], 'best_epoch': state.get('best_epoch'),
              'best_loss': state.get('best_loss'), 'epoch': state.get('epoch'),
              'optimizer_updates': state.get('optimizer_updates')}
    (output / (name + '.input-verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def build_eval_worker(config):
    """Generate a credential-free worker using the unchanged, pinned evaluator."""
    required = ('job_id', 'variant', 'seed', 'source_revision', 'source_kernel',
                'checkpoint_sha256', 'job_sha256')
    payload = {key: config[key] for key in required}
    if not re.fullmatch(r'[a-z_]+_seed\d+', payload['job_id']):
        raise ValueError('Invalid job identity')
    for key in ('checkpoint_sha256', 'job_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', payload[key]):
            raise ValueError('Invalid input hash')
    if not re.fullmatch(r'[0-9a-f]{40}', payload['source_revision']):
        raise ValueError('Invalid source revision')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/\d+', payload['source_kernel']):
        raise ValueError('Source kernel must have a pinned version')
    verifier = inspect.getsource(prepare_eval_inputs)
    inner = verifier + '''
import json, os, subprocess, sys, time, runpy
from pathlib import Path
from scripts.prepare_kaggle_datasets import find_mounted_kaggle_dataset
from scripts.run_v3 import validate_job, validate_resume
import torch

config = json.loads(sys.argv[1])
work = Path('/kaggle/working')
proof = prepare_eval_inputs(config, Path('/kaggle/input'), work)
job = json.loads(Path(proof['job_file']).read_text())
validate_job(job)
state = torch.load(proof['checkpoint'], map_location='cpu', weights_only=True)
validate_resume(state, job)
del state
if not torch.cuda.is_available():
    raise RuntimeError('A GPU runtime is required for full evaluation')
print('V3_EVAL_INPUT_VERIFIED ' + json.dumps(proof), flush=True)
data_root = find_mounted_kaggle_dataset()
if data_root is None:
    raise RuntimeError('Real held-out MOT17 data is required')
torch.set_num_threads(2)
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
sys.argv = ['scripts/run_v3.py', '--mode', 'eval', '--job-file', proof['job_file'],
            '--checkpoint', proof['checkpoint'], '--data-root', str(data_root),
            '--output-dir', str(work), '--eval-yolo']
runpy.run_path('scripts/run_v3.py', run_name='__main__')
torch.cuda.synchronize()
runtime = {'job_id': config['job_id'], 'seconds': time.monotonic()-started,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
(work / (config['job_id'] + '.runtime.json')).write_text(json.dumps(runtime, indent=2))
print('V3_EVAL_COMPLETE ' + json.dumps(runtime), flush=True)
'''
    return '''"""Private, revision-pinned V3 held-out evaluation worker."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
CONFIG = json.loads(''' + repr(json.dumps(payload)) + ''')
INNER = ''' + repr(inner) + '''
repo = Path(tempfile.gettempdir())/'tscope-eval-source'
subprocess.run(['git','clone','https://github.com/munnn01/t_scope_v3.git',str(repo)], check=True)
subprocess.run(['git','checkout','--detach',CONFIG['source_revision']], cwd=repo, check=True)
actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
if actual != CONFIG['source_revision'] or subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True).strip():
    raise RuntimeError('Pinned evaluation source mismatch')
subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.9'],check=True)
subprocess.run([sys.executable,'-m','uv','python','install','3.11.14'],check=True)
runtime = Path(tempfile.gettempdir())/'tscope-eval-env'
subprocess.run([sys.executable,'-m','uv','venv','--python','3.11.14',str(runtime)],check=True)
python = str(runtime/'bin/python')
subprocess.run([sys.executable,'-m','uv','pip','install','--python',python,'-r',str(repo/'requirements-eval.txt')],check=True)
environment = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='2')
with (Path('/kaggle/working')/(CONFIG['job_id']+'.environment.txt')).open('w') as handle:
    subprocess.run([sys.executable,'-m','uv','pip','freeze','--python',python], stdout=handle, check=True)
subprocess.run([python,'-u','-c',INNER,json.dumps(CONFIG)],cwd=repo,env=environment,check=True)
'''
