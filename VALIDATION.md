# Release validation

Checks were performed on 2026-10-27 with Python 3.12.12, PyTorch 2.4.0,
torchvision 0.19.0, and transformers 4.42.3. Offline checks disabled CUDA.
No robot motion, camera acquisition, or reward-server requests were made.

## Packaging and tests

- Built and installed all three wheels in an isolated directory.
- Checked source distributions for required scripts, configuration, tokenizer
  assets, licenses, and the Deoxys compatibility patch.
- Passed 29 tests: 19 world-model/data tests, 5 reward tests, and 5 control tests.
- Checked training, evaluation, preprocessing, reward, and control CLI help.
- Checked both world-model shell wrappers against the installed argument
  parsers and verified the active recipe and held-out split.
- Confirmed that the world-model distribution excludes the archived trainer,
  experimental cross-attention model, and alternate trajectory-ID evaluator.

The tests cover training/evaluation defaults, temporal alignment, held-out
splits, action gradients, normalization, reward pooling and phase reversal,
policy target mapping, control presets, and mocked Robometer requests.

Dataset-export tests cover lossless values and dtypes, source preservation,
resume behavior, altered-output rejection, parallel export, terminal
observations, and controller-configuration attributes. A regression test
checks evaluation output paths for both consolidated and episode-file inputs.

## Existing checkpoint compatibility

- The selected `wm_kstep_concat_h1_p1_fs8_iter67000.pth` loaded with strict
  state-dict matching. A CPU forward produced `(1, 1, 196, 768)` embeddings
  for each RGB camera and `(1, 1, 196, 512)` tactile embeddings.
- The tactile contrastive reward produced a finite, nonzero gradient with
  respect to a proposed eight-action block.
- The four-frame AnyTouch checkpoint loaded all required tactile and text
  weights. A forward produced tactile tokens of shape `(1, 398, 512)` and
  text embeddings of shape `(2, 512)`.
- The existing `itr_23999.ckpt` diffusion policy loaded strictly with the
  bundled backend. Two CPU denoising steps produced finite `(1, 16, 7)` actions.

## Real-episode execution check

An exported pipette episode was evaluated on CPU with the selected 67k
checkpoint and both existing decoders. The evaluator wrote a nine-frame MP4
and three finite camera-embedding arrays. The first eight predicted-array
entries matched copied ground-truth context; the last entry was a prediction.

The reward README's NPZ-scoring example then produced one finite contrastive
reward row from that prediction, using the existing AnyTouch checkpoint.
This checks the dataset → world model → reward interface with one transition;
it does not measure model quality.

## Dataset and source preservation

All three dataset exports passed lossless verification: 980 episodes,
302,631 action frames, and 716.51 GB of episode files. The audit confirmed
unchanged source identities and normalization files, original trajectory IDs,
seed-42 splits, terminal observations, and controller metadata.
See [DATASETS.md](DATASETS.md).

The original research files were preserved. SHA-256 checks covered all 43
source files selected during cleanup, including the three variants retained
only in the research archive. The GitHub release includes 40 of those source
files; per-package manifests also record generated files.

Whitespace cleanup in the GitHub copy preserved every Python module's parsed
syntax tree. Compatibility patch contents retain their original whitespace.

The gripper patch was applied only to an isolated copy of the original base
source, producing the existing modified source byte for byte. The live Deoxys
checkout was not patched.

## Limits

These checks establish packaging, data preservation, interfaces, and
checkpoint compatibility. They do not reproduce full training convergence,
paper metrics, or robot success rates.

The normalizer rejects a supplied missing file instead of silently using
generic statistics. Using the correct training statistics may change results
relative to old runs with the stale normalization path.
