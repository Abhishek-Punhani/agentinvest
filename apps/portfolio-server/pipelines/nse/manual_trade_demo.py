#!/usr/bin/env python3
"""
Manual Trade Creation Demo - Uses existing TradeController
NO Celery, NO async processing - just direct trade submission for demo
"""

import os
import sys
import asyncio
import time

# Ensure environment variables are loaded (DATABASE_URL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except Exception:
    pass  # best effort

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db import DBManager
from controllers.trade_controller import TradeController
from schemas import TradeRequest

# Add Kafka publisher for notifications
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SHARED_PY_PATH = PROJECT_ROOT / "shared" / "py"
if str(SHARED_PY_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_PY_PATH))

from kafka_service import default_kafka_bus, PublisherAlreadyRegistered
from pydantic import BaseModel
from datetime import datetime


# Demo signals
DEMO_SIGNALS = [
    {
        "symbol": "GRASIM",
        "action": "BUY",
        "confidence": 0.95,
        "reason": "BlackRock subsidiary investing ₹3,000 crore - major positive catalyst",
        "price": 2650.50,
        "attachment_url": "https://nsearchives.nseindia.com/corporate/GRASIM_09122025214838_SEIntimationGIP.pdf",
        "subject": "General Updates - BlackRock Investment",
    },
    {
        "symbol": "INDIGO",
        "action": "SELL",
        "confidence": 0.95,
        "reason": "DGCA orders 10% capacity reduction due to operational mismanagement",
        "price": 4250.75,
        "attachment_url": "https://nsearchives.nseindia.com/corporate/Indigo1_09122025213248_DGCA_notice_09122025.pdf",
        "subject": "General Updates - DGCA Capacity Reduction",
    },
    {
        "symbol": "ZAGGLE",
        "action": "BUY",
        "confidence": 0.95,
        "reason": "5-year strategic agreement with Mastercard - significant partnership",
        "price": 385.20,
        "attachment_url": "https://nsearchives.nseindia.com/corporate/ZAGGLE_09122025223918_MasterCard.pdf",
        "subject": "Bagging/Receiving of orders/contracts - Mastercard Agreement",
    },
    {
        "symbol": "DHARAN",
        "action": "SELL",
        "confidence": 0.98,
        "reason": "Auditor warns 'going concern' uncertainty - critical negative catalyst",
        "price": 142.30,
        "attachment_url": "https://nsearchives.nseindia.com/corporate/KBCGLOBAL_09122025235654_results300625.pdf",
        "subject": "Outcome of Board Meeting - Going Concern Warning",
    },
]


def print_banner(text, char="═"):
    """Print a banner"""
    print(f"\n{char * 80}")
    print(text)
    print(f"{char * 80}\n")


def print_signal(idx, total, signal):
    """Print signal details"""
    print("─" * 80)
    print(f"📊 SIGNAL {idx}/{total}")
    print("─" * 80)
    print(f"Symbol:     {signal['symbol']}")
    print(f"Action:     {'🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL'}")
    print(f"Price:      ₹{signal['price']:.2f}")
    print(f"Confidence: {signal['confidence']*100:.0f}%")
    print(f"Reason:     {signal['reason']}")
    print("─" * 80)


# Kafka signal schema matching NSE pipeline
class NSESignalEvent(BaseModel):
    symbol: str
    filing_time: str
    signal: int
    explanation: str
    confidence: float
    generated_at: str
    source: str = "manual_demo"
    subject_of_announcement: str = ""
    attachment_url: str = ""
    date_time_of_submission: str = ""


def get_kafka_publisher():
    """Get or create Kafka publisher for trading signals"""
    KAFKA_SIGNAL_TOPIC = "nse_filings_trading_signal"
    KAFKA_PUBLISHER_NAME = "manual_demo_signal_publisher"
    
    bus = default_kafka_bus
    try:
        publisher = bus.register_publisher(
            KAFKA_PUBLISHER_NAME,
            topic=KAFKA_SIGNAL_TOPIC,
            value_model=NSESignalEvent,
            default_headers={"stream": "manual_demo"},
        )
    except PublisherAlreadyRegistered:
        publisher = bus.get_publisher(KAFKA_PUBLISHER_NAME)
    
    return publisher


def publish_signal_to_kafka(signal_data: dict) -> None:
    """Publish trading signal to Kafka to trigger notifications"""
    try:
        publisher = get_kafka_publisher()
        event = NSESignalEvent(**signal_data)
        publisher.publish(event.model_dump(), key=signal_data["symbol"])
        print(f"[KAFKA] ✅ Published signal to Kafka for {signal_data['symbol']}")
    except Exception as e:
        print(f"[KAFKA] ⚠️ Failed to publish to Kafka: {e}")


def calculate_allocation(confidence: float, capital: float = 100000.0) -> float:
    """Calculate allocation based on confidence"""
    if confidence >= 0.95:
        return capital * 0.20  # 20% for very high confidence
    elif confidence >= 0.90:
        return capital * 0.15  # 15% for high confidence
    elif confidence >= 0.85:
        return capital * 0.10  # 10% for medium confidence
    else:
        return capital * 0.05  # 5% for low confidence


async def create_trade_via_controller(controller, portfolio_id, agent_id, allocation_id, org_id, customer_id, signal, price):
    """Use TradeController to create trade"""
    
    # Calculate quantity based on allocation
    available_capital = 100000.0  # Demo capital
    allocated_amount = calculate_allocation(signal['confidence'], available_capital)
    quantity = int(allocated_amount / float(price))
    
    if quantity <= 0:
        print(f"❌ Quantity is 0 for {signal['symbol']}, skipping")
        return None
    
    # Build trade request with short selling support
    # Use "SHORT_SELL" side for SELL orders (bypasses position check)
    side_value = signal['action'].upper()
    if side_value == "SELL":
        side_value = "SHORT_SELL"  # Enable short selling for SELL signals
    
    trade_request = TradeRequest(
        portfolio_id=portfolio_id,
        allocation_id=allocation_id,
        symbol=signal['symbol'],
        side=side_value,  # Use SHORT_SELL for sells without positions
        order_type="market",  # lowercase for Pydantic validation
        quantity=quantity,
        exchange="NSE",
        segment="EQ",
    )
    
    # Create mock request and user
    class MockRequest:
        def __init__(self):
            self.headers = {}
            self.client = type('obj', (object,), {'host': 'localhost'})
    
    mock_user = {
        "id": customer_id,
        "customer_id": customer_id,
        "organization_id": org_id,
        "role": "admin",
    }
    
    # Submit trade via controller
    result = await controller.submit_trade(trade_request, MockRequest(), mock_user)
    return result


async def main():
    print_banner("🎬 MANUAL TRADE DEMO - Using TradeController", "═")
    print(f"📊 Total Signals: {len(DEMO_SIGNALS)}")
    print(f"⏱️  Pause between trades: 2 seconds")
    print("═" * 80)
    
    from db import DBManager
    db_manager = DBManager.get_instance()
    await db_manager.connect()
    client = db_manager.get_client()
    
    try:
        # Get first active high_risk agent
        print("\n🔍 Finding active high_risk trading agent...")
        agents = await client.tradingagent.find_many(
            where={
                "agent_type": "high_risk",
                "status": "active",
            },
            include={
                "portfolio": True,
                "allocation": True,
            },
            take=1,
        )
        
        if not agents:
            print("❌ No active high_risk agent found!")
            print("   Create one in Prisma Studio first")
            return
        
        agent = agents[0]
        print(f"✅ Using agent: {agent.id}")
        print(f"   Portfolio: {agent.portfolio.id}")
        capital_val = getattr(agent.allocation, "allocated_amount", 100000.0)
        print(f"   Capital: ₹{capital_val:,.2f}")

        # Get IDs from portfolio
        portfolio_row = await client.portfolio.find_unique(
            where={"id": agent.portfolio.id}
        )
        org_id = portfolio_row.organization_id if portfolio_row else None
        customer_id = portfolio_row.customer_id if portfolio_row else None
        allocation_id = getattr(agent, "portfolio_allocation_id", None)
        
        if not org_id or not customer_id or not allocation_id:
            print(f"❌ Missing IDs - org:{org_id}, customer:{customer_id}, allocation:{allocation_id}")
            return
            
        print(f"   Org: {org_id}")
        print(f"   Customer: {customer_id}")
        print(f"   Allocation: {allocation_id}")
        
        # Initialize TradeController
        controller = TradeController(client)
        
        # Process each signal
        successful = 0
        failed = 0
        
        for idx, signal in enumerate(DEMO_SIGNALS, 1):
            print_signal(idx, len(DEMO_SIGNALS), signal)
            
            try:
                # Publish signal to Kafka FIRST (for notifications)
                kafka_signal = {
                    "symbol": signal['symbol'],
                    "filing_time": datetime.utcnow().isoformat() + "Z",
                    "signal": 1 if signal['action'] == 'BUY' else -1,
                    "explanation": signal['reason'],
                    "confidence": signal['confidence'],
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "attachment_url": signal.get('attachment_url', ''),
                    "subject_of_announcement": signal.get('subject', ''),
                }
                publish_signal_to_kafka(kafka_signal)
                
                # Wait 1 second after publishing signal (let notification system process)
                print("⏳ Waiting 1 second for signal processing...")
                time.sleep(1)
                
                # Then execute trade via controller
                result = await create_trade_via_controller(
                    controller, 
                    agent.portfolio.id,
                    agent.id,
                    allocation_id,
                    org_id,
                    customer_id,
                    signal,
                    signal['price']
                )
                
                if result and result.success:
                    # Extract first trade from trades list
                    if result.trades and len(result.trades) > 0:
                        trade = result.trades[0]
                        print(f"✅ Trade submitted: {trade.id}")
                        print(f"   Symbol: {trade.symbol}")
                        print(f"   Side: {trade.side}")
                        print(f"   Quantity: {trade.quantity}")
                        print(f"   Status: {trade.status}")
                        if trade.price:
                            print(f"   Price: ₹{trade.price}")
                        successful += 1
                    else:
                        print(f"✅ {result.message}")
                        successful += 1
                else:
                    print(f"⚠️ Trade submission failed: {result.message if result else 'Unknown error'}")
                    failed += 1
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed to submit trade: {error_msg}")
                import traceback
                traceback.print_exc()
                failed += 1
            
            print(f"\n✅ Signal {idx} processed\n")
            
            if idx < len(DEMO_SIGNALS):
                print("📸 SCREENSHOT TIME")
                print("   Check Prisma Studio: Trades and Positions tables")
                print("\n⏳ Waiting 2 seconds before next trade...")
                for i in range(2, 0, -1):
                    print(f"   {i}...")
                    time.sleep(1)
        
        print_banner("🎬 DEMO COMPLETE", "═")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed:     {failed}")
        print(f"📊 Total:      {len(DEMO_SIGNALS)}")
        print("═" * 80)
        print("\n💡 Check Prisma Studio to see all trades and positions!")
        print("   Trades table: All trades with execution details")
        print("   Positions table: Open positions from BUY orders")
        print("═" * 80)
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
