VENV := pydemucs
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install export compare onnx_separate clean

help:
	@echo "Targets:"
	@echo "  make venv      - create venv $(VENV)"
	@echo "  make install   - install deps into $(VENV)"
	@echo "  make export    - export htdemucs core to ONNX"
	@echo "  make compare   - compare PyTorch vs ONNX on test.mp3"
	@echo "  make onnx_separate INPUT=... [ONNX=...] [OUT=...] [NAME=...] [EXT=wav]"
	@echo "  make clean     - remove ONNX artifacts"

venv:
	python3 -m venv $(VENV)
	$(PY) -V

install: venv
	# PyTorch CPU wheels for Python 3.12
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchaudio==2.4.1
	# Project deps and ONNX runtime
	$(PIP) install dora-search einops julius lameenc openunmix pyyaml tqdm onnx onnxruntime soundfile

export: install
	$(PY) -m tools.export_onnx -n htdemucs -o htdemucs_core.onnx --opset 17 --dynamic

compare: export
	$(PY) -m tools.compare_onnx -n htdemucs -m htdemucs_core.onnx -i test.mp3 --sr 44100

# Example:
# make onnx_separate INPUT=your.mp3 OUT=separated_onnx ONNX=htdemucs_core.onnx NAME=htdemucs EXT=wav
onnx_separate:
	@if [ -z "$(INPUT)" ]; then echo "Usage: make onnx_separate INPUT=your.mp3 [ONNX=htdemucs_core.onnx] [OUT=separated_onnx] [NAME=htdemucs] [EXT=wav]"; exit 1; fi
	$(PY) -m tools.simple_separate_onnx $(INPUT) --onnx $${ONNX:-htdemucs_core.onnx} --out $${OUT:-separated_onnx} -n $${NAME:-htdemucs} --ext $${EXT:-wav}

clean:
	rm -f htdemucs_core.onnx
all: linter tests

linter:
	flake8 demucs
	mypy demucs

tests: test_train test_eval

test_train: tests/musdb
	_DORA_TEST_PATH=/tmp/demucs python3 -m dora run --clear \
		dset.musdb=./tests/musdb dset.segment=4 dset.shift=2 epochs=2 model=demucs \
		demucs.depth=2 demucs.channels=4 test.sdr=false misc.num_workers=0 test.workers=0 \
		test.shifts=0

test_eval:
	python3 -m demucs -n demucs_unittest test.mp3
	python3 -m demucs -n demucs_unittest --two-stems=vocals test.mp3
	python3 -m demucs -n demucs_unittest --mp3 test.mp3
	python3 -m demucs -n demucs_unittest --flac --int24 test.mp3
	python3 -m demucs -n demucs_unittest --int24 --clip-mode clamp test.mp3
	python3 -m demucs -n demucs_unittest --segment 8 test.mp3
	python3 -m demucs.api -n demucs_unittest --segment 8 test.mp3
	python3 -m demucs --list-models

tests/musdb:
	test -e tests || mkdir tests
	python3 -c 'import musdb; musdb.DB("tests/tmp", download=True)'
	musdbconvert tests/tmp tests/musdb

dist:
	python3 setup.py sdist

clean:
	rm -r dist build *.egg-info

.PHONY: linter dist test_train test_eval
