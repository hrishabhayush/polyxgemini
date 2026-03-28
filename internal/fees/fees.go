package fees

import "math"

// PolymarketSportsTakerFee returns the taker fee for 1 share at the given price.
// Formula: fee = p × 0.0175 × (p × (1-p))
// where p = share price (0 to 1).
func PolymarketSportsTakerFee(price float64) float64 {
	if price <= 0 || price >= 1 {
		return 0
	}
	return price * 0.0175 * price * (1 - price)
}

// PolymarketCryptoTakerFee returns the taker fee for 1 share at the given price.
// Formula: fee = p × 0.25 × (p × (1-p))^2
func PolymarketCryptoTakerFee(price float64) float64 {
	if price <= 0 || price >= 1 {
		return 0
	}
	pq := price * (1 - price)
	return price * 0.25 * pq * pq
}

// GeminiTakerFee returns the taker fee for 1 share at the given price.
// Formula: fee = 0.07 × p × (1-p)
// Fees are rounded up to the next cent.
func GeminiTakerFee(price float64) float64 {
	if price <= 0 || price >= 1 {
		return 0
	}
	fee := 0.07 * price * (1 - price)
	// Round up to next cent (0.01)
	return math.Ceil(fee*100) / 100
}

// EffectiveBuyCost returns the total cost to buy 1 share including fee.
func EffectiveBuyCost(askPrice, feePerShare float64) float64 {
	return askPrice + feePerShare
}
