.PHONY: build run test fmt lint test-sentiment finbert-server test-finbert-python

build:
	go build -o bin/bot ./cmd/bot

run:
	go run ./cmd/bot

test:
	go test ./...

fmt:
	go fmt ./...

lint:
	golangci-lint run ./...

test-sentiment:
	go test -v -tags=integration -run . ./internal/sentiment/...

finbert-server:
	cd python/finbert && uvicorn server:app --host 127.0.0.1 --port 8765

test-finbert-python:
	cd python/finbert && python test_finbert.py
