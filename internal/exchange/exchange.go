package exchange

import "context"

// Exchange is the common interface both Polymarket and Gemini clients implement.
type Exchange interface {
	FetchMarkets(ctx context.Context) ([]Market, error)
	FetchOrderBook(ctx context.Context, marketID string) (*OrderBook, error)
	PlaceOrder(ctx context.Context, order *OrderRequest) (*Order, error)
	CancelOrder(ctx context.Context, orderID string) error
}

type Market struct {
	ID       string
	Name     string
	Outcomes []Outcome
}

type Outcome struct {
	ID    string
	Label string
	Price float64
}

type OrderBook struct {
	MarketID string
	Bids     []PriceLevel
	Asks     []PriceLevel
}

type PriceLevel struct {
	Price    float64
	Quantity float64
}

type OrderRequest struct {
	MarketID string
	Side     string // "buy" or "sell"
	Outcome  string // "yes" or "no"
	Price    float64
	Quantity float64
}

type Order struct {
	ID       string
	Status   string
	FilledQty float64
}
