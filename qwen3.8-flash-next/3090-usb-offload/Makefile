# make        -> ./out (build bundle: fatbins, patched Strata, configure tree)    same as: docker compose run --rm build
# make fetch  -> ./out/models (~130 GB incl. the 84 GB dense GGUF; EXPERT_LAYERS="24" for one layer)
# make run    -> ./run.sh (native, on the Mac)
# make test   -> scripts/test-gates.sh (card-free regression check of the verification gates)
JOBS ?= 4
.PHONY: all build fetch run check test clean
all: build
build:
	docker buildx build --platform linux/arm64 --target bundle --build-arg JOBS=$(JOBS) --output type=local,dest=out .
fetch:
	docker compose --profile fetch run --rm fetch all
run:
	./run.sh
check:
	./run.sh --check
test:
	bash scripts/test-gates.sh
clean:
	rm -rf out
