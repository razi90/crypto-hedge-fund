use std::str::FromStr;
use radix_common::address::AddressBech32Decoder;
use radix_common::network::NetworkDefinition;
use radix_common::types::ComponentAddress;
use radix_common_derive::dec;
use radix_transactions::manifest::decompile;
use radix_transactions::model::SubintentManifestV2;
use anthic_client::AnthicClient;
use anthic_model::{AnthicAddressInfo, AnthicConfig};
use anthic_subintents::*;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Anthic API key
    #[arg(long)]
    api_key: String,

    /// User account address
    #[arg(long)]
    account: String,

    /// Buy token symbol (e.g., "Test-xwBTC")
    #[arg(long)]
    buy_symbol: String,

    /// Buy token amount
    #[arg(long)]
    buy_amount: String,

    /// Sell token symbol (e.g., "Test-xUSDC")
    #[arg(long)]
    sell_symbol: String,

    /// Sell token amount
    #[arg(long)]
    sell_amount: String,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let network = NetworkDefinition::from_str("stokenet").unwrap();
    let trade_api_url = "https://trade-api.staging.anthic.io";

    let user_address = {
        let decoder = AddressBech32Decoder::new(&network);
        ComponentAddress::try_from_bech32(&decoder, &cli.account).unwrap()
    };

    // A high level Anthic client which wraps calls to the Anthic API
    let client = AnthicClient::new(
        network.clone(),
        trade_api_url.to_string(),
        cli.api_key
    );

    // Anthic configuration
    let anthic_config = client.load_anthic_config().await.unwrap();

    // Address info
    let address_info = client.load_account_address_info(user_address).await.unwrap();

    // Create buy/sell token amounts from CLI args
    let buy = TokenAmount {
        symbol: cli.buy_symbol,
        amount: dec!(&cli.buy_amount),
    };
    let sell = TokenAmount {
        symbol: cli.sell_symbol,
        amount: dec!(&cli.sell_amount),
    };

    // Create the manifest for the order
    let manifest = create_order_manifest(&anthic_config, user_address, &address_info, sell, buy).unwrap();

    println!("{}", decompile(&manifest, &network).unwrap());
}

fn create_order_manifest(
    anthic_config: &AnthicConfig,
    account_address: ComponentAddress,
    address_info: &AnthicAddressInfo,
    sell: TokenAmount,
    buy: TokenAmount
) -> Result<SubintentManifestV2, String> {
    let builder = AnthicSubintentManifestBuilder::new(anthic_config.clone());

    // There is a flat solver fee which includes transaction execution fee, a portion of which will be rebated
    // in the transaction
    let solver_fee_amount = anthic_config.settlement_fee_per_resource.get(&sell.symbol).unwrap().clone();
    let anthic_fee_percent = anthic_config.anthic_fee_per_level.get(address_info.level as usize).unwrap().taker_fee.clone();
    let anthic_fee_amount = sell.amount * anthic_fee_percent;

    let manifest = builder.add_anthic_limit_order(account_address, sell, buy, solver_fee_amount, anthic_fee_amount).build();
    Ok(manifest)
}