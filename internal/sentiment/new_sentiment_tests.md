
go test -v -tags=integration ./internal/sentiment/gdelt/  
NEWSAPI_KEY=8a4028324fe348a6a6b337f09287562f go test -v -tags=integration ./internal/sentiment/newsapi/
go test -v -tags=integration ./internal/sentiment/reddit/ 

make finbert-server
go test -v -tags=integration ./internal/sentiment/finbert/

//hardcoded comments:
go test ./internal/sentiment/gamma/gamma_test/... -run "TestFilterRelevant|TestPositionWeight" -v

//live gamma comments:
go test ./internal/sentiment/gamma/gamma_test/... -run TestFetchGammaComments -v