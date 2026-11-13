.PHONY: install test smoke benchmark stream profile

install:
	python -m pip install -e '.[test]'

test:
	pytest -q

smoke:
	framerate-lab benchmark --device cpu --weights none --batch-sizes 1 --precisions fp32 --warmup 1 --iterations 2

benchmark:
	framerate-lab benchmark --device auto --batch-sizes 1,2,4,8 --precisions fp32,fp16 --warmup 20 --iterations 100

stream:
	framerate-lab stream --device auto --target-fps 30 --duration 10 --queue-size 4

profile:
	framerate-lab profile --device auto --batch-size 1 --precision fp32
