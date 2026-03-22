package executor

import (
	"github.com/hrishabhayush/polyxgemini/internal/engine"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Executor struct {
	poly   exchange.Exchange
	gemini exchange.Exchange
	dryRun bool
}

func NewExecutor(poly, gemini exchange.Exchange, dryRun bool) *Executor {
	return &Executor{poly: poly, gemini: gemini, dryRun: dryRun}
}

func (ex *Executor) Execute(opp engine.Opportunity) error {
	// TODO: Place buy order on one exchange, sell on the other
	// TODO: If dryRun, just log the opportunity
	return nil
}
