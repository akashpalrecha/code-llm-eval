# There are two dockerfiles: for all benchmarks, and for MultiPL-E
DOCKERFILE=Dockerfile

ifeq ($(DOCKERFILE), Dockerfile)
	IMAGE_NAME=evaluation-harness
else
	IMAGE_NAME=evaluation-harness-multiple
endif

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

all: build