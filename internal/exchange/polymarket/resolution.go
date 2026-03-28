package polymarket

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// SourceMeta holds resolution source metadata across all enrichment tiers.
type SourceMeta struct {
	RawSource        string  // original resolutionSource from Gamma
	Domain           string  // extracted domain (e.g. "reuters.com")
	IsURL            bool    // whether RawSource parses as a valid URL
	ReliabilityScore float64 // 0.0–1.0 domain-based reliability
	EnrichmentStatus string  // "ok", "unavailable", "not_url", "timeout", "non_html"
	ExtractedTitle   string  // page <title> if fetched
	ExtractedSnippet string  // first ~500 chars of body text
}

// ClassifySource performs Tier 1-2 enrichment: parses the raw resolution source
// string, extracts the domain, and assigns a reliability score.
func ClassifySource(rawSource string) SourceMeta {
	meta := SourceMeta{RawSource: rawSource}

	if rawSource == "" {
		meta.EnrichmentStatus = "no_source"
		return meta
	}

	u, err := url.Parse(strings.TrimSpace(rawSource))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		meta.EnrichmentStatus = "not_url"
		meta.ReliabilityScore = 0.2
		return meta
	}

	meta.IsURL = true
	meta.Domain = strings.ToLower(u.Hostname())
	meta.ReliabilityScore = DomainReliabilityScore(meta.Domain)
	meta.EnrichmentStatus = "classified"
	return meta
}

// domainScores maps known domains to reliability scores.
var domainScores = map[string]float64{
	// Tier 1: official / government / authoritative data feeds
	"bls.gov": 0.95, "sec.gov": 0.95, "treasury.gov": 0.95,
	"federalreserve.gov": 0.95, "whitehouse.gov": 0.95,
	"congress.gov": 0.95, "supremecourt.gov": 0.95,

	// Tier 1: major wire services / sports data
	"apnews.com": 0.93, "reuters.com": 0.93,
	"espn.com": 0.92, "sports-reference.com": 0.90,
	"mlb.com": 0.90, "nba.com": 0.90, "nfl.com": 0.90,
	"ncaa.com": 0.90, "ncaa.org": 0.90,

	// Tier 2: major global outlets
	"nytimes.com": 0.85, "washingtonpost.com": 0.85,
	"bbc.com": 0.85, "bbc.co.uk": 0.85,
	"bloomberg.com": 0.85, "ft.com": 0.85,
	"wsj.com": 0.85, "economist.com": 0.83,
	"cnn.com": 0.80, "foxnews.com": 0.75,

	// Tier 3: general news / reputable sources
	"theguardian.com": 0.78, "usatoday.com": 0.75,
	"cbssports.com": 0.78, "foxsports.com": 0.75,
	"yahoo.com": 0.70, "cnbc.com": 0.78,

	// Crypto / prediction market specific
	"coingecko.com": 0.75, "coinmarketcap.com": 0.75,
	"defillama.com": 0.72, "dune.com": 0.70,
	"etherscan.io": 0.80, "polygonscan.com": 0.78,
}

// DomainReliabilityScore returns a reliability score for a given domain.
// Known authoritative domains score 0.7-0.95; unknown domains default to 0.3.
func DomainReliabilityScore(domain string) float64 {
	domain = strings.ToLower(domain)

	if score, ok := domainScores[domain]; ok {
		return score
	}

	// Check parent domain (e.g. "api.espn.com" → "espn.com")
	parts := strings.Split(domain, ".")
	if len(parts) > 2 {
		parent := strings.Join(parts[len(parts)-2:], ".")
		if score, ok := domainScores[parent]; ok {
			return score
		}
	}

	// Government domains are generally trustworthy
	if strings.HasSuffix(domain, ".gov") || strings.HasSuffix(domain, ".gov.uk") {
		return 0.90
	}
	if strings.HasSuffix(domain, ".edu") {
		return 0.80
	}
	if strings.HasSuffix(domain, ".org") {
		return 0.60
	}

	return 0.30
}

// EnrichSource performs Tier 3: best-effort HTTP fetch of the resolution source
// URL and extracts the page title and a text snippet. Fails gracefully.
func EnrichSource(ctx context.Context, meta *SourceMeta) {
	if !meta.IsURL || meta.RawSource == "" {
		return
	}

	fetchCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(fetchCtx, http.MethodGet, meta.RawSource, nil)
	if err != nil {
		meta.EnrichmentStatus = "unavailable"
		return
	}
	req.Header.Set("User-Agent", "polymarket-report/1.0")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		meta.EnrichmentStatus = "timeout"
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		meta.EnrichmentStatus = fmt.Sprintf("http_%d", resp.StatusCode)
		return
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/html") && !strings.Contains(ct, "text/plain") {
		meta.EnrichmentStatus = "non_html"
		return
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 50_000))
	if err != nil {
		meta.EnrichmentStatus = "read_error"
		return
	}
	html := string(body)

	meta.ExtractedTitle = extractTitle(html)
	meta.ExtractedSnippet = extractTextSnippet(html, 500)
	meta.EnrichmentStatus = "ok"
}

// extractTitle pulls content from the first <title>...</title> tag.
func extractTitle(html string) string {
	lower := strings.ToLower(html)
	start := strings.Index(lower, "<title")
	if start == -1 {
		return ""
	}
	gt := strings.Index(lower[start:], ">")
	if gt == -1 {
		return ""
	}
	contentStart := start + gt + 1
	end := strings.Index(lower[contentStart:], "</title>")
	if end == -1 {
		return ""
	}
	title := strings.TrimSpace(html[contentStart : contentStart+end])
	if len(title) > 200 {
		title = title[:200]
	}
	return title
}

// extractTextSnippet strips HTML tags and returns the first maxLen characters
// of visible text content.
func extractTextSnippet(html string, maxLen int) string {
	// Remove script and style blocks
	for _, tag := range []string{"script", "style", "noscript"} {
		for {
			lower := strings.ToLower(html)
			start := strings.Index(lower, "<"+tag)
			if start == -1 {
				break
			}
			end := strings.Index(lower[start:], "</"+tag+">")
			if end == -1 {
				break
			}
			html = html[:start] + html[start+end+len("</"+tag+">"):]
		}
	}

	var b strings.Builder
	inTag := false
	for _, r := range html {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			b.WriteRune(' ')
			continue
		}
		if !inTag {
			b.WriteRune(r)
		}
	}

	text := strings.Join(strings.Fields(b.String()), " ")
	if len(text) > maxLen {
		text = text[:maxLen]
	}
	return strings.TrimSpace(text)
}
