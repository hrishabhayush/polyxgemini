package budget

import (
	"context"
	"log"
	"math"
	"sync"
)

// Budget tracks cumulative spend across all markets and exchanges.
// When the limit is reached, it cancels the provided context to stop the bot.
type Budget struct {
	mu     sync.Mutex
	spent  float64
	limit  float64
	cancel context.CancelFunc
}

// New creates a Budget with the given spending limit and context cancel func.
func New(limit float64, cancel context.CancelFunc) *Budget {
	log.Printf("[BUDGET] initialized: limit=$%.2f", limit)
	return &Budget{
		limit:  limit,
		cancel: cancel,
	}
}

// TrySpend attempts to spend up to amount from the budget.
// Returns the actual amount spent and whether any spend occurred.
// If the remaining budget is less than amount, it spends only what's left.
// Returns (0, false) if the budget is already exhausted.
func (b *Budget) TrySpend(amount float64) (float64, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	remaining := b.limit - b.spent
	if remaining <= 0 {
		return 0, false
	}

	actual := math.Min(amount, remaining)
	b.spent += actual

	if b.spent >= b.limit {
		log.Printf("[BUDGET] $%.2f limit reached (spent $%.4f), shutting down", b.limit, b.spent)
		b.cancel()
	}

	return actual, true
}

// Remaining returns how much budget is left.
func (b *Budget) Remaining() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	r := b.limit - b.spent
	if r < 0 {
		return 0
	}
	return r
}

// Spent returns the total amount spent so far.
func (b *Budget) Spent() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.spent
}

// Exhausted returns true if the budget is fully spent.
func (b *Budget) Exhausted() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.spent >= b.limit
}
