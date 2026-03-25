.PHONY: build run test fmt lint dashboard-up dashboard-down dashboard-logs

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

dashboard-up:
	docker compose up -d

dashboard-down:
	docker compose down

dashboard-logs:
	docker compose logs -f
