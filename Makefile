.PHONY: build run test fmt lint test-sentiment finbert-server test-finbert-python export snapshot ml-server dashboard-up dashboard-down dashboard-logs

build:
	go build -o bin/bot ./cmd/bot
	go build -o bin/export ./cmd/export
	go build -o bin/snapshot ./cmd/snapshot

run:
	go run ./cmd/bot

export:
	go run ./cmd/export

snapshot:
	go run ./cmd/snapshot

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

ml-server:
	cd python/ml && uvicorn serve:app --host 127.0.0.1 --port 8766

test-finbert-python:
	cd python/finbert && python test_finbert.py

dashboard-up:
	docker compose up -d

dashboard-down:
	docker compose down

dashboard-logs:
	docker compose logs -f
