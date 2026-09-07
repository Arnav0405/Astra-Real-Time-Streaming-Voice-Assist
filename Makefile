GO_DIR := services/backend
ML_DIR := services/ml

# Guard Go targets until first .go file exists — go tools error on empty modules
GO_FILES := $(shell find $(GO_DIR) -name '*.go' 2>/dev/null | head -1)

.PHONY: format lint test proto tts-voice clean docker up

format:
ifneq ($(GO_FILES),)
	cd $(GO_DIR) && gofmt -w .
endif
	cd $(ML_DIR) && uv run ruff format .

lint:
ifneq ($(GO_FILES),)
	cd $(GO_DIR) && go vet ./...
endif
	cd $(ML_DIR) && uv run ruff check .

test:
ifneq ($(GO_FILES),)
	cd $(GO_DIR) && go test ./...
endif
	cd $(ML_DIR) && uv run pytest; status=$$?; test $$status -eq 0 -o $$status -eq 5

# Requires protoc + protoc-gen-go (go install google.golang.org/protobuf/cmd/protoc-gen-go@latest)
proto:
	PATH="$$PATH:$$(go env GOPATH)/bin" protoc \
		--proto_path=proto \
		--go_out=$(GO_DIR) \
		--go_opt=module=github.com/arnav/astra/services/backend \
		proto/astra/v1/astra.proto \
		proto/astra/v1/asr.proto \
		proto/astra/v1/tts.proto

docker:
	docker build -f docker/Dockerfile -t astra .

up:
	docker compose up --build

tts-voice:
	python3 scripts/fetch_tts_voice.py

clean:
	rm -rf $(GO_DIR)/bin $(ML_DIR)/.pytest_cache $(ML_DIR)/.ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
