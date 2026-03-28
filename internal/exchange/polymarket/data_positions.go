package polymarket

import (
	"sort"
)

// WalletVolume summarizes a wallet's total trading volume in a market.
type WalletVolume struct {
	Wallet    string
	Volume    float64
	SharePct  float64 // fraction of total volume (0–100)
}

// WalletConcentration returns the top-N wallets by trading volume and the HHI
// derived from the trade distribution in the provided trades slice.
//
// Polymarket does not expose a public market-wide positions endpoint, so wallet
// trading volume is used as a proxy for position concentration.
//
// ⚠ Limitation: a single trader may split activity across multiple wallets,
// which will underestimate true concentration. Treat high HHI as a strong
// manipulation-risk signal, but low HHI as only moderate reassurance.
func WalletConcentration(trades []Trade, topN int) (hhi float64, top []WalletVolume) {
	volByWallet := make(map[string]float64)
	var total float64

	for _, t := range trades {
		if t.ProxyWallet == "" {
			continue
		}
		volByWallet[t.ProxyWallet] += t.Size
		total += t.Size
	}

	if total == 0 {
		return 0, nil
	}

	// Build sorted slice
	all := make([]WalletVolume, 0, len(volByWallet))
	for wallet, vol := range volByWallet {
		share := vol / total
		hhi += share * share
		all = append(all, WalletVolume{
			Wallet:   wallet,
			Volume:   vol,
			SharePct: share * 100,
		})
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].Volume > all[j].Volume
	})

	if topN > len(all) {
		topN = len(all)
	}
	return hhi, all[:topN]
}

// HHI computes the Herfindahl-Hirschman Index from a slice of WalletVolume entries.
// Exposed separately in case the caller already has WalletVolume data.
//
//	HHI = Σ (volume_i / total)²
//
// Range: 0.0 (fully distributed) → 1.0 (one wallet dominates).
func HHI(wallets []WalletVolume) float64 {
	var total float64
	for _, w := range wallets {
		total += w.Volume
	}
	if total == 0 {
		return 0
	}
	var hhi float64
	for _, w := range wallets {
		share := w.Volume / total
		hhi += share * share
	}
	return hhi
}
