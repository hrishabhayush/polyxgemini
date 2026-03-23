package sentiment

import "time"

// Label mirrors FinBERT output labels.
type Label string

const (
	Positive Label = "positive"
	Negative Label = "negative"
	Neutral  Label = "neutral"
)

// ArticleScore is the raw FinBERT output for one piece of text.
type ArticleScore struct {
	Text      string
	Label     Label
	Score     float64
	Source    string
	FetchedAt time.Time
}

// MarketSignal is the aggregated rolling sentiment for one market.
type MarketSignal struct {
	MarketID     string
	BullishScore float64
	BearishScore float64
	ArticleCount int
	LastUpdated  time.Time
}

// Aggregator maintains a sliding window of ArticleScores per market.
type Aggregator struct {
	windowSize int
	signals    map[string][]ArticleScore
}

func NewAggregator(windowSize int) *Aggregator {
	return &Aggregator{
		windowSize: windowSize,
		signals:    make(map[string][]ArticleScore),
	}
}

// Add appends a scored article to the market's window, evicting the oldest if full.
func (a *Aggregator) Add(marketID string, score ArticleScore) {
	window := a.signals[marketID]
	window = append(window, score)
	if len(window) > a.windowSize {
		window = window[len(window)-a.windowSize:]
	}
	a.signals[marketID] = window
}

// Signal computes the current MarketSignal for a market.
// BullishScore = avg confidence of positive articles; BearishScore for negative.
func (a *Aggregator) Signal(marketID string) MarketSignal {
	window := a.signals[marketID]
	if len(window) == 0 {
		return MarketSignal{MarketID: marketID}
	}

	var posSum, negSum float64
	var posCount, negCount int
	for _, s := range window {
		switch s.Label {
		case Positive:
			posSum += s.Score
			posCount++
		case Negative:
			negSum += s.Score
			negCount++
		}
	}

	sig := MarketSignal{
		MarketID:     marketID,
		ArticleCount: len(window),
		LastUpdated:  window[len(window)-1].FetchedAt,
	}
	if posCount > 0 {
		sig.BullishScore = posSum / float64(posCount)
	}
	if negCount > 0 {
		sig.BearishScore = negSum / float64(negCount)
	}
	return sig
}
