package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

const (
	polygonChainID = 137
	clobBaseURL    = "https://clob.polymarket.com"
	msgToSign      = "This message attests that I control the given wallet"
)

// EIP-712 type hashes — domain has NO verifyingContract, timestamp is string
var (
	domainTypeHash = crypto.Keccak256Hash([]byte(
		"EIP712Domain(string name,string version,uint256 chainId)",
	))
	clobAuthTypeHash = crypto.Keccak256Hash([]byte(
		"ClobAuth(address address,string timestamp,uint256 nonce,string message)",
	))
)

func clobAuthDomainSeparator() common.Hash {
	nameHash := crypto.Keccak256Hash([]byte("ClobAuthDomain"))
	versionHash := crypto.Keccak256Hash([]byte("1"))
	chainID := new(big.Int).SetInt64(polygonChainID)

	data := make([]byte, 0, 128)
	data = append(data, domainTypeHash.Bytes()...)
	data = append(data, nameHash.Bytes()...)
	data = append(data, versionHash.Bytes()...)
	data = append(data, common.LeftPadBytes(chainID.Bytes(), 32)...)
	return crypto.Keccak256Hash(data)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: derivekey <private_key_hex>\n")
		fmt.Fprintf(os.Stderr, "  Derives a Polymarket API key tied to the EOA address of the given private key.\n")
		os.Exit(1)
	}

	pkHex := strings.TrimPrefix(os.Args[1], "0x")
	privateKey, err := crypto.HexToECDSA(pkHex)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid private key: %v\n", err)
		os.Exit(1)
	}

	address := crypto.PubkeyToAddress(privateKey.PublicKey)
	fmt.Printf("EOA address: %s\n", address.Hex())

	timestamp := time.Now().Unix()
	nonce := rand.Int63()

	tsStr := strconv.FormatInt(timestamp, 10)
	nonceStr := strconv.FormatInt(nonce, 10)

	// EIP-712 struct hash for ClobAuth
	// string fields are keccak256-hashed per EIP-712 spec
	tsHash := crypto.Keccak256Hash([]byte(tsStr))
	msgHash := crypto.Keccak256Hash([]byte(msgToSign))
	nonceBig := new(big.Int).SetInt64(nonce)

	structData := make([]byte, 0, 32*5)
	structData = append(structData, clobAuthTypeHash.Bytes()...)
	structData = append(structData, common.LeftPadBytes(address.Bytes(), 32)...)
	structData = append(structData, tsHash.Bytes()...)
	structData = append(structData, common.LeftPadBytes(nonceBig.Bytes(), 32)...)
	structData = append(structData, msgHash.Bytes()...)
	sHash := crypto.Keccak256Hash(structData)

	domain := clobAuthDomainSeparator()

	// EIP-712 digest: "\x19\x01" || domainSeparator || structHash
	digest := crypto.Keccak256Hash(
		append(append([]byte{0x19, 0x01}, domain.Bytes()...), sHash.Bytes()...),
	)

	sig, err := crypto.Sign(digest.Bytes(), privateKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "signing failed: %v\n", err)
		os.Exit(1)
	}
	if sig[64] < 27 {
		sig[64] += 27
	}
	sigHex := fmt.Sprintf("0x%x", sig)

	setHeaders := func(req *http.Request) {
		req.Header["POLY_ADDRESS"] = []string{address.Hex()}
		req.Header["POLY_SIGNATURE"] = []string{sigHex}
		req.Header["POLY_TIMESTAMP"] = []string{tsStr}
		req.Header["POLY_NONCE"] = []string{nonceStr}
	}

	// Try POST /auth/api-key first (create), fall back to GET /auth/derive-api-key (retrieve)
	createURL := clobBaseURL + "/auth/api-key"
	fmt.Printf("Creating API key via POST %s ...\n", createURL)
	req, err := http.NewRequest(http.MethodPost, createURL, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "request error: %v\n", err)
		os.Exit(1)
	}
	setHeaders(req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "HTTP error: %v\n", err)
		os.Exit(1)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Create returned %d: %s — trying derive...\n", resp.StatusCode, string(body))
		deriveURL := clobBaseURL + "/auth/derive-api-key"
		req, err = http.NewRequest(http.MethodGet, deriveURL, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "request error: %v\n", err)
			os.Exit(1)
		}
		setHeaders(req)

		resp, err = http.DefaultClient.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "HTTP error: %v\n", err)
			os.Exit(1)
		}
		body, _ = io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Derive also failed (status %d): %s\n", resp.StatusCode, string(body))
			os.Exit(1)
		}
	}

	var creds struct {
		APIKey     string `json:"apiKey"`
		Secret     string `json:"secret"`
		Passphrase string `json:"passphrase"`
	}
	if err := json.Unmarshal(body, &creds); err != nil {
		fmt.Fprintf(os.Stderr, "parse error: %v\nraw: %s\n", err, string(body))
		os.Exit(1)
	}

	fmt.Println("\n=== New Polymarket API Credentials ===")
	fmt.Printf("api_key:    %s\n", creds.APIKey)
	fmt.Printf("api_secret: %s\n", creds.Secret)
	fmt.Printf("passphrase: %s\n", creds.Passphrase)
	fmt.Println("\nPaste these into config/config.yaml under the polymarket section.")
	fmt.Printf("Also set wallet_address to: %s\n", address.Hex())
}
