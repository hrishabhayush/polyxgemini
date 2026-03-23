
go test -v -tags=integration ./internal/sentiment/gdelt/  
NEWSAPI_KEY=8a4028324fe348a6a6b337f09287562f go test -v -tags=integration ./internal/sentiment/newsapi/
go test -v -tags=integration ./internal/sentiment/reddit/ 

make finbert-server
go test -v -tags=integration ./internal/sentiment/finbert/