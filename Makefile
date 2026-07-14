GO_DIR := services/backend
ML_DIR := services/ml

# Guard Go targets until first .go file exists — go tools error on empty modules
GO_FILES := $(shell find $(GO_DIR) -name '*.go' 2>/dev/null | head -1)

.PHONY: format lint test proto clean

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

# ponytail: placeholder until proto/ has schemas — then wire protoc + protoc-gen-go
proto:
	@echo "proto: no schemas in proto/ yet — nothing to generate"

clean:
	rm -rf $(GO_DIR)/bin $(ML_DIR)/.pytest_cache $(ML_DIR)/.ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
