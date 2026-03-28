package polymarket

import (
	"crypto/ecdsa"
	"fmt"
	"math/big"
	"math/rand"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

const (
	SideBuy  = 0
	SideSell = 1

	SigTypeEOA        = 0
	SigTypePolyProxy  = 1
	SigTypeGnosisSafe = 2

	// Polymarket CTF Exchange on Polygon mainnet (non-negRisk)
	ExchangeAddress    = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
	NegRiskExchange    = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
	ZeroAddress        = "0x0000000000000000000000000000000000000000"

	PolygonChainID = 137
)

// orderTypeHash is keccak256 of the Order EIP-712 type string.
var orderTypeHash = crypto.Keccak256Hash([]byte(
	"Order(uint256 salt,address maker,address signer,address taker,uint256 tokenId,uint256 makerAmount,uint256 takerAmount,uint256 expiration,uint256 nonce,uint256 feeRateBps,uint8 side,uint8 signatureType)",
))

func domainSeparator(exchange string) common.Hash {
	nameHash := crypto.Keccak256Hash([]byte("Polymarket CTF Exchange"))
	versionHash := crypto.Keccak256Hash([]byte("1"))
	chainID := new(big.Int).SetInt64(PolygonChainID)

	// EIP-712 domain: keccak256(typeHash || nameHash || versionHash || chainId || verifyingContract)
	domainTypeHash := crypto.Keccak256Hash([]byte(
		"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
	))

	data := make([]byte, 0, 160)
	data = append(data, domainTypeHash.Bytes()...)
	data = append(data, nameHash.Bytes()...)
	data = append(data, versionHash.Bytes()...)
	data = append(data, common.LeftPadBytes(chainID.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(common.HexToAddress(exchange).Bytes(), 32)...)

	return crypto.Keccak256Hash(data)
}

type SignedOrderPayload struct {
	Order     map[string]any `json:"order"`
	Owner     string         `json:"owner"`
	OrderType string         `json:"orderType"`
	PostOnly  bool           `json:"postOnly"`
}

type orderFields struct {
	Salt          *big.Int
	Maker         common.Address
	Signer        common.Address
	Taker         common.Address
	TokenID       *big.Int
	MakerAmount   *big.Int
	TakerAmount   *big.Int
	Expiration    *big.Int
	Nonce         *big.Int
	FeeRateBPS    *big.Int
	Side          uint8
	SignatureType uint8
}

func generateSalt() *big.Int {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	now := time.Now().Unix()
	salt := float64(now) * r.Float64()
	return new(big.Int).SetInt64(int64(salt))
}

func structHash(o *orderFields) common.Hash {
	data := make([]byte, 0, 32*13)
	data = append(data, orderTypeHash.Bytes()...)
	data = append(data, common.LeftPadBytes(o.Salt.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.Maker.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.Signer.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.Taker.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.TokenID.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.MakerAmount.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.TakerAmount.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.Expiration.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.Nonce.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes(o.FeeRateBPS.Bytes(), 32)...)
	data = append(data, common.LeftPadBytes([]byte{o.Side}, 32)...)
	data = append(data, common.LeftPadBytes([]byte{o.SignatureType}, 32)...)
	return crypto.Keccak256Hash(data)
}

func signOrder(key *ecdsa.PrivateKey, o *orderFields, negRisk bool) (string, error) {
	exchange := ExchangeAddress
	if negRisk {
		exchange = NegRiskExchange
	}
	domain := domainSeparator(exchange)
	sHash := structHash(o)

	// EIP-712 message: "\x19\x01" || domainSeparator || structHash
	msg := make([]byte, 0, 66)
	msg = append(msg, 0x19, 0x01)
	msg = append(msg, domain.Bytes()...)
	msg = append(msg, sHash.Bytes()...)

	digest := crypto.Keccak256Hash(msg)
	sig, err := crypto.Sign(digest.Bytes(), key)
	if err != nil {
		return "", fmt.Errorf("sign order: %w", err)
	}

	// go-ethereum returns [R || S || V] where V is 0 or 1; EIP-155 needs 27/28
	if sig[64] < 27 {
		sig[64] += 27
	}

	return fmt.Sprintf("0x%x", sig), nil
}

// BuildSignedOrder creates the full signed order payload ready for the /order endpoint.
func BuildSignedOrder(
	privateKey *ecdsa.PrivateKey,
	funderAddress string,
	signerAddress string,
	tokenID string,
	side int,
	makerAmount *big.Int,
	takerAmount *big.Int,
	feeRateBPS int,
	sigType int,
	apiKey string,
	orderType string,
	negRisk bool,
) (*SignedOrderPayload, error) {
	tokenIDBig, ok := new(big.Int).SetString(tokenID, 10)
	if !ok {
		tokenIDBig, ok = new(big.Int).SetString(tokenID, 0)
		if !ok {
			return nil, fmt.Errorf("invalid tokenID: %s", tokenID)
		}
	}

	o := &orderFields{
		Salt:          generateSalt(),
		Maker:         common.HexToAddress(funderAddress),
		Signer:        common.HexToAddress(signerAddress),
		Taker:         common.HexToAddress(ZeroAddress),
		TokenID:       tokenIDBig,
		MakerAmount:   makerAmount,
		TakerAmount:   takerAmount,
		Expiration:    big.NewInt(0),
		Nonce:         big.NewInt(0),
		FeeRateBPS:    big.NewInt(int64(feeRateBPS)),
		Side:          uint8(side),
		SignatureType: uint8(sigType),
	}

	signature, err := signOrder(privateKey, o, negRisk)
	if err != nil {
		return nil, err
	}

	sideStr := "BUY"
	if side == SideSell {
		sideStr = "SELL"
	}

	return &SignedOrderPayload{
		Order: map[string]any{
			"salt":          o.Salt.Int64(),
			"maker":         funderAddress,
			"signer":        signerAddress,
			"taker":         ZeroAddress,
			"tokenId":       tokenID,
			"makerAmount":   makerAmount.String(),
			"takerAmount":   takerAmount.String(),
			"expiration":    "0",
			"nonce":         "0",
			"feeRateBps":    fmt.Sprintf("%d", feeRateBPS),
			"side":          sideStr,
			"signatureType": sigType,
			"signature":     signature,
		},
		Owner:     apiKey,
		OrderType: orderType,
	}, nil
}

// ComputeAmounts converts price and size into USDC-denominated makerAmount and takerAmount.
// Polymarket requires makerAmount to have max 2 USDC decimal places (multiple of 10000)
// and takerAmount to have max 4 USDC decimal places (multiple of 100).
func ComputeAmounts(side int, price, size float64) (makerAmount, takerAmount *big.Int) {
	scale := new(big.Float).SetFloat64(1e6)

	if side == SideBuy {
		maker := new(big.Float).SetFloat64(size * price)
		maker.Mul(maker, scale)
		taker := new(big.Float).SetFloat64(size)
		taker.Mul(taker, scale)

		makerAmount, _ = maker.Int(nil)
		takerAmount, _ = taker.Int(nil)
	} else {
		maker := new(big.Float).SetFloat64(size)
		maker.Mul(maker, scale)
		taker := new(big.Float).SetFloat64(size * price)
		taker.Mul(taker, scale)

		makerAmount, _ = maker.Int(nil)
		takerAmount, _ = taker.Int(nil)
	}

	makerAmount = roundDown(makerAmount, 10000)
	takerAmount = roundDown(takerAmount, 100)
	return
}

func roundDown(v *big.Int, unit int64) *big.Int {
	u := big.NewInt(unit)
	v.Div(v, u)
	v.Mul(v, u)
	return v
}
