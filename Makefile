.PHONY: build run test fmt lint

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
