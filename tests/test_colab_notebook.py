import json
from pathlib import Path



def test_colab_setup_preserves_preinstalled_cuda_pytorch():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])

    assert "'--no-deps','-e','.'" in setup
    assert "'pip','install','--upgrade'" not in setup
    assert "from packaging.version import Version" in setup
    assert "Version('2.10.0')" in setup
    assert "pyarrow>=15.0" in setup
    assert "pip','install','-e','.[dev,colab]'" not in setup


def test_colab_expensive_stages_fail_closed_without_gpu_and_preflight_memory():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    five_m_stage = "".join(notebook["cells"][8]["source"])
    long_stage = "".join(notebook["cells"][10]["source"])
    sft_stage = "".join(notebook["cells"][12]["source"])

    assert "if not torch.cuda.is_available():" in five_m_stage
    assert "if not torch.cuda.is_available():" in long_stage
    assert "torch.cuda.mem_get_info(0)" in long_stage
    assert "LONG_MIN_FREE_GPU_GIB = 8.0" in long_stage
    assert "if not torch.cuda.is_available():" in sft_stage
    assert "torch.cuda.mem_get_info(0)" in sft_stage
    assert "LONG_TARGET_TOKENS_OVERRIDE must be a positive integer or None." in long_stage

def test_colab_long_training_has_no_fixed_optimizer_step_cap():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    long_stage = "".join(notebook["cells"][10]["source"])

    assert "LONG_MAX_OPTIMIZER_STEPS" not in long_stage
    assert "LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "BASE_STEPS = LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "if LONG_OPTIMIZER_STEPS_OVERRIDE is None:" in long_stage
    assert "if TARGET_TOKENS < TOKENS_PER_OPTIMIZER_STEP:" in long_stage
    assert "No arbitrary maximum optimizer-step cap is applied." in long_stage


def test_colab_training_targets_do_not_change_checkpoint_run_identity():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    long_stage = "".join(notebook["cells"][10]["source"])
    long_identity = long_stage.split("base_identity = json.dumps({", 1)[1].split("BASE_KEY =", 1)[0]
    assert "'steps': BASE_STEPS" not in long_identity
    assert "'target_tokens': ACTUAL_TOKENS" not in long_identity
    assert "args.append('--extend-schedule')" in long_stage

    five_m_stage = "".join(notebook["cells"][8]["source"])
    five_m_identity = five_m_stage.split("identity_5m = {", 1)[1].split("KEY_5M =", 1)[0]
    assert "'steps': INTERMEDIATE_5M_STEPS" not in five_m_identity
    assert "ARGS_5M.append('--extend-schedule')" in five_m_stage

    sft_stage = "".join(notebook["cells"][12]["source"])
    sft_identity = sft_stage.split("json.dumps({", 1)[1].split("SFT_KEY =", 1)[0]
    assert "'steps': SFT_STEPS" not in sft_identity
    assert "sft_args.append('--extend-schedule')" in sft_stage

def test_sft_never_silently_selects_the_tiny_smoke_checkpoint():
    notebook_path = Path(__file__).resolve().parents[1] / 'colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][12]['source'])
    assert 'SFT_SOURCE_OVERRIDE = None' in sft_stage
    assert 'SFT_MINIMUM_SOURCE_PARAMETERS = 50_000_000' in sft_stage
    assert 'base_*/final_model/model_manifest.json' in sft_stage
    assert 'qmini-final-model-v2' in sft_stage
    assert "SFT_SOURCE = SMOKE/'final_model'" not in sft_stage
    assert 'The tiny smoke model is never selected silently.' in sft_stage
    assert sft_stage.index('SFT_SOURCE = select_sft_source()') > sft_stage.index('if not sft_files:')


def test_sft_requires_independent_task_and_conversation_holdouts():
    notebook_path = Path(__file__).resolve().parents[1] / 'colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][12]['source'])
    gate = sft_stage.index('if not SFT_HELDOUT_TASKS.is_file() or not SFT_HELDOUT_CONVERSATIONS.is_file():')
    training_run = sft_stage.index("SFT_RUN = ART/'runs'/f'sft_{SFT_KEY}'")
    identity = sft_stage.index("sft_identity.update(SFT_HELDOUT_TASKS_SHA256.encode('utf-8'))")
    assert gate < training_run
    assert identity < sft_stage.index('SFT_KEY =')
    assert 'training_file_hashes.intersection(heldout_hashes)' in sft_stage



def test_colab_runs_random_brain_architecture_preflight_before_dataset_loading():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])
    preflight = "".join(notebook["cells"][3]["source"])
    tokenizer_data_stage = "".join(notebook["cells"][4]["source"])

    assert "qmini.cli','preflight'" in preflight
    assert "qmini_research_152m.json" in preflight
    assert "'--sequence-length','8'" in preflight
    assert "if torch.cuda.is_available():" in preflight
    assert "qmini_research_152m.json" in preflight
    assert "qmini_smoke.json" in preflight
    assert "PREFLIGHT_DEVICE = 'cpu'" in preflight
    assert "'--device',PREFLIGHT_DEVICE" in preflight
    assert "optimizer step" in preflight
    assert "weights updated" in preflight
    import ast
    ast.parse(preflight)
    assert "subprocess.run(ARCHITECTURE_PREFLIGHT,check=True)" in preflight
    assert "qmini.cli','tokenizer'" in tokenizer_data_stage
    assert setup.index("pip','install','--no-deps','-e','.'") < len(setup)

def test_colab_smoke_learning_proof_loads_training_checkpoint_not_dense_snapshot():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    smoke_stage = "".join(notebook["cells"][6]["source"])

    assert "checkpoint_payload = torch.load(checkpoint_path" in smoke_stage
    assert "checkpoint_payload.get('format') != 'qmini-checkpoint-v2'" in smoke_stage
    assert "trained_state = checkpoint_payload.get('model')" in smoke_stage
    assert "SMOKE/'final_model'/'model.pt'" not in smoke_stage
    assert "minimum_nll_improvement': 0.05" in smoke_stage
    assert "before_report['cross_entropy'] - after_report['cross_entropy'] >= 0.05" in smoke_stage
    assert "if not checkpoint_path.is_file():" in smoke_stage
    assert smoke_stage.index("if not checkpoint_path.is_file():") < smoke_stage.index(
        "checkpoint_payload = torch.load(checkpoint_path"
    )

def test_training_ready_builder_bundles_dormant_training_runtime_and_never_starts_it():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "colab"
        / "QMini_Training_Ready_Build.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    import ast
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{notebook_path}:code-cell-{index}")

    settings = code_cells[0]
    build = code_cells[4]
    final_guard = code_cells[5]
    assert 'MODEL_CONFIG_REL = "configs/qmini_research_152m.json"' in settings
    assert "RUN_TRAINING_NOW" not in settings
    assert "build-initial" in build
    assert "verify_packed_export" in build
    assert "_verify_initial_brain_artifact" in build
    assert "_verify_prepared_dataset_manifest" in build
    assert "training_runtime" in build
    assert '"enabled": False' in build
    assert "training_progress.json" in build
    assert "training_runtime_enabled" in build
    assert "training_runtime_manifest_sha256" in build
    assert 'model_manifest.get("training_runtime_manifest_sha256") != sha256_file(runtime_manifest_path)' in build
    assert "QMini_Training_Runner.ipynb" in final_guard
    assert "train_model(" not in build
    markdown = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "QMini_Training_Runner.ipynb" in markdown


def test_training_ready_package_identity_includes_preparation_contract():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "colab"
        / "QMini_Training_Ready_Build.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    identity_cell = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "package_identity =" in "".join(cell.get("source", []))
    )
    assert '"loss_mask_mode": LOSS_MASK_MODE' in identity_cell
    assert '"minimum_sequence_length": TRAIN_SEQUENCE_LENGTH' in identity_cell
def test_training_runner_is_separate_and_has_verified_resume_controls():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training_Runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    import ast
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{notebook_path}:code-cell-{index}")
    settings = code_cells[0]
    run = "".join(code_cells)
    assert "START_TRAINING = False" in settings
    assert "restore_latest_checkpoint" in run
    assert "TrainingSessionLease" in run
    assert "checkpoint_mirror_dir" in run
    assert "checkpoint_session_id" in run
    assert "mark_training_status" in run
    assert "model.qmini.npz" in run
    assert "training_runtime_manifest_sha256" in run
    assert "No checkpoint restore or optimizer update has run in this cell." in code_cells[3]
    assert "restore_latest_checkpoint" not in code_cells[3]
    activation = next(
        source for source in code_cells
        if "with TrainingSessionLease(" in source and "restore_latest_checkpoint" in source
    )
    assert activation.index("with TrainingSessionLease(") < activation.index("restore_latest_checkpoint")
    assert activation.index("restore_latest_checkpoint") < activation.index("train_model(")


