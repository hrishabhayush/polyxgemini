package polymarket

import "testing"

func TestResolveOutcomeFromGamma(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		m    gammaMarketResponse
		want string
	}{
		{
			name: "singular_outcome_field",
			m: gammaMarketResponse{
				Closed:  true,
				Outcome: "Yes",
			},
			want: "Yes",
		},
		{
			name: "active_no_infer",
			m: gammaMarketResponse{
				Closed:        false,
				Outcomes:      `["Yes", "No"]`,
				OutcomePrices: `["0.2", "0.8"]`,
			},
			want: "",
		},
		{
			name: "closed_binary_yes_wins",
			m: gammaMarketResponse{
				Closed:        true,
				Outcomes:      `["Yes", "No"]`,
				OutcomePrices: `["1", "0"]`,
			},
			want: "Yes",
		},
		{
			name: "closed_binary_no_wins",
			m: gammaMarketResponse{
				Closed:        true,
				Outcomes:      `["Yes", "No"]`,
				OutcomePrices: `["0", "1"]`,
			},
			want: "No",
		},
		{
			name: "closed_all_zero",
			m: gammaMarketResponse{
				Closed:        true,
				Outcomes:      `["Yes", "No"]`,
				OutcomePrices: `["0", "0"]`,
			},
			want: "",
		},
		{
			name: "closed_float_one",
			m: gammaMarketResponse{
				Closed:        true,
				Outcomes:      `["Yes", "No"]`,
				OutcomePrices: `["1.0", "0.0"]`,
			},
			want: "Yes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := resolveOutcomeFromGamma(tt.m)
			if got != tt.want {
				t.Errorf("resolveOutcomeFromGamma() = %q, want %q", got, tt.want)
			}
		})
	}
}
