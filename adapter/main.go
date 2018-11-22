package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/big"
	"strings"

	ethereum "github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/singnet/reputation/adapter/resources/contracts/mpe"
)

//ChannelOpenEvent event struct
type ChannelOpenEvent struct {
	ChannelID  *big.Int
	Sender     common.Address
	Recipient  common.Address
	GroupID    [32]byte
	Signer     common.Address
	Amount     *big.Int
	Expiration *big.Int
}

//ChannelClaimEvent event struct
type ChannelClaimEvent struct {
	ChannelID      *big.Int
	Recipient      common.Address
	ClaimAmount    *big.Int
	SendBackAmount *big.Int
	KeepAmount     *big.Int
}

//ChannelSenderClaimEvent event struct
type ChannelSenderClaimEvent struct {
	ChannelID   *big.Int
	ClaimAmount *big.Int
}

//ChannelExtend event struct
type ChannelExtend struct {
	ChannelID     *big.Int
	NewExpiration *big.Int
}

//ChannelAddFunds event struct
type ChannelAddFunds struct {
	ChannelID *big.Int
	NewFunds  *big.Int
}

//Network config struct
type Network struct {
	RPCEndpoint     string
	DeployedAddress common.Address
	startingBlock   int64
}

var networks = map[string]Network{
	"kovan": Network{
		"https://kovan.infura.io",
		common.HexToAddress("0x385036D6cd8Cf6A8749d5Df7f716F0341E1c13B1"),
		0,
	},
	"ropsten": Network{
		"https://ropsten.infura.io",
		common.HexToAddress("0xAF5e3b8CF89815F24A12D45D4758D87257249778"),
		4429391,
	},
}

func main() {
	networkKey := flag.String("network", "ropsten", "network. One of {mainnet, ropsten, kovan}")
	currentNetwork := networks[*networkKey]

	client, err := ethclient.Dial(currentNetwork.RPCEndpoint)
	if err != nil {
		log.Fatal(err)
	}

	query := ethereum.FilterQuery{
		FromBlock: big.NewInt(currentNetwork.startingBlock),
		Addresses: []common.Address{currentNetwork.DeployedAddress},
	}

	logs, err := client.FilterLogs(context.Background(), query)
	if err != nil {
		log.Fatal(err)
	}

	mpeAbi, err := abi.JSON(strings.NewReader(string(mpe.MpeABI)))
	if err != nil {
		log.Fatal(err)
	}

	for _, vLog := range logs {
		fmt.Println(vLog.TxHash.Hex())
		channelOpenEvent := ChannelOpenEvent{}
		err := mpeAbi.Unpack(&channelOpenEvent, "ChannelOpen", vLog.Data)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(channelOpenEvent.ChannelID)
	}

}
