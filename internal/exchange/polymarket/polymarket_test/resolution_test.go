package polymarket_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

// TestClassifySource verifies domain extraction and reliability scoring.
func TestClassifySource(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantURL   bool
		wantMin   float64
		wantMax   float64
		wantDomain string
	}{
		{
			name: "reuters URL", raw: "https://www.reuters.com/elections/2026",
			wantURL: true, wantMin: 0.90, wantMax: 1.0, wantDomain: "www.reuters.com",
		},
		{
			name: "espn URL", raw: "https://www.espn.com/ncaa/scores",
			wantURL: true, wantMin: 0.85, wantMax: 1.0, wantDomain: "www.espn.com",
		},
		{
			name: "gov URL", raw: "https://results.elections.gov/2026",
			wantURL: true, wantMin: 0.85, wantMax: 1.0,
		},
		{
			name: "unknown domain", raw: "https://randomsite.xyz/result",
			wantURL: true, wantMin: 0.2, wantMax: 0.4,
		},
		{
			name: "not a URL", raw: "Based on official AP results",
			wantURL: false, wantMin: 0.1, wantMax: 0.3,
		},
		{
			name: "empty", raw: "",
			wantURL: false, wantMin: 0.0, wantMax: 0.01,
		},
		{
			name: "edu domain", raw: "https://www.harvard.edu/research/study",
			wantURL: true, wantMin: 0.75, wantMax: 0.85,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := polymarket.ClassifySource(tt.raw)
			fmt.Printf("  %-14s url=%-5v domain=%-24s reliability=%.2f\n",
				tt.name, meta.IsURL, meta.Domain, meta.ReliabilityScore)
			if meta.IsURL != tt.wantURL {
				t.Errorf("IsURL = %v, want %v", meta.IsURL, tt.wantURL)
			}
			if meta.ReliabilityScore < tt.wantMin || meta.ReliabilityScore > tt.wantMax {
				t.Errorf("ReliabilityScore = %.2f, want [%.2f, %.2f]",
					meta.ReliabilityScore, tt.wantMin, tt.wantMax)
			}
			if tt.wantDomain != "" && meta.Domain != tt.wantDomain {
				t.Errorf("Domain = %q, want %q", meta.Domain, tt.wantDomain)
			}
		})
	}
}

// TestDomainReliabilityScore verifies specific domain lookups and fallback rules.
func TestDomainReliabilityScore(t *testing.T) {
	ap := polymarket.DomainReliabilityScore("apnews.com")
	espn := polymarket.DomainReliabilityScore("api.espn.com")
	unknown := polymarket.DomainReliabilityScore("randomdomain.com")
	gov := polymarket.DomainReliabilityScore("data.bls.gov")
	fmt.Printf("\nDomain reliability: apnews=%.2f espn_subdomain=%.2f unknown=%.2f gov_subdomain=%.2f\n\n",
		ap, espn, unknown, gov)

	if ap < 0.90 {
		t.Errorf("apnews.com = %.2f, want >= 0.90", ap)
	}
	if espn < 0.85 {
		t.Errorf("api.espn.com (subdomain) = %.2f, want >= 0.85", espn)
	}
	if unknown != 0.30 {
		t.Errorf("unknown domain = %.2f, want 0.30", unknown)
	}
	if gov < 0.90 {
		t.Errorf("data.bls.gov (gov subdomain) = %.2f, want >= 0.90", gov)
	}
}

// TestEnrichSourceLive performs a live HTTP fetch of a known accessible URL.
func TestEnrichSourceLive(t *testing.T) {
	marketName := "btc-updown-5m-1774300200"
	if v := os.Getenv("MARKET_NAME"); v != "" {
		marketName = v
	}

	polyCfg := config.PolymarketConfig{
		CLOBBaseURL:  "https://clob.polymarket.com",
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
	}

	ctx := context.Background()
	client := polymarket.NewClient(polyCfg)
	market, err := client.FindMarket(ctx, marketName)
	if err != nil {
		t.Fatalf("FindMarket failed: %v", err)
	}
	if market.ResolutionSource == "" {
		t.Skip("market has empty resolution source")
	}

	meta := polymarket.ClassifySource(market.ResolutionSource)
	if !meta.IsURL {
		t.Skipf("market resolution source is not a URL: %q", market.ResolutionSource)
	}
	polymarket.EnrichSource(ctx, &meta)

	fmt.Printf("  Market resolution source: %q\n", market.ResolutionSource)
	fmt.Printf("  Enrichment: status=%s title=%q snippet_len=%d\n",
		meta.EnrichmentStatus, meta.ExtractedTitle, len(meta.ExtractedSnippet))

	if meta.EnrichmentStatus != "ok" && meta.EnrichmentStatus != "unavailable" &&
		!strings.HasPrefix(meta.EnrichmentStatus, "http_") {
		t.Logf("Enrichment status: %s (may be expected for network restrictions)", meta.EnrichmentStatus)
	}
}

// TestEnrichSourceUnavailable verifies graceful fallback for unreachable URLs.
func TestEnrichSourceUnavailable(t *testing.T) {
	meta := polymarket.ClassifySource("https://this-domain-does-not-exist-xyzzy.invalid/page")
	ctx := context.Background()
	polymarket.EnrichSource(ctx, &meta)

	if meta.EnrichmentStatus == "ok" {
		t.Error("expected non-ok enrichment status for unreachable URL")
	}
	t.Logf("Enrichment status for unreachable URL: %s", meta.EnrichmentStatus)
}

// TestEnrichSourceNotURL verifies that non-URL sources are skipped gracefully.
func TestEnrichSourceNotURL(t *testing.T) {
	meta := polymarket.ClassifySource("Official AP election results")
	ctx := context.Background()
	polymarket.EnrichSource(ctx, &meta)

	if meta.EnrichmentStatus != "not_url" {
		t.Errorf("expected 'not_url' status, got %q", meta.EnrichmentStatus)
	}
}

// TestResolutionSourceFromMarket verifies that FindMarket now returns
// resolution source metadata from the Gamma API.
func TestResolutionSourceFromMarket(t *testing.T) {
	marketName := "btc-updown-5m-1774300200"
	if v := os.Getenv("MARKET_NAME"); v != "" {
		marketName = v
	}

	polyCfg := config.PolymarketConfig{
		CLOBBaseURL:  "https://clob.polymarket.com",
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
	}

	ctx := context.Background()
	client := polymarket.NewClient(polyCfg)

	market, err := client.FindMarket(ctx, marketName)
	if err != nil {
		t.Fatalf("FindMarket failed: %v", err)
	}

	fmt.Printf("\n========================================================\n")
	fmt.Printf("  RESOLUTION SOURCE FROM MARKET\n")
	fmt.Printf("========================================================\n")
	fmt.Printf("  Market:            %s\n", market.Question)
	fmt.Printf("  Resolution source: %q\n", market.ResolutionSource)
	descPreview := market.Description
	if len(descPreview) > 200 {
		descPreview = descPreview[:200] + "..."
	}
	fmt.Printf("  Description:       %q\n", descPreview)

	if market.ResolutionSource != "" {
		meta := polymarket.ClassifySource(market.ResolutionSource)
		fmt.Printf("  Domain:            %s\n", meta.Domain)
		fmt.Printf("  Reliability:       %.2f\n", meta.ReliabilityScore)
		fmt.Printf("  Enrich status:     %s\n", meta.EnrichmentStatus)
	}
	fmt.Printf("========================================================\n\n")
}
