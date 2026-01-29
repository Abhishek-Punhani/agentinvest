#!/usr/bin/env python3
"""
NSE Pipeline Demo - Step-by-step signal publishing and trade execution.
Shows the full flow: Signal → Kafka → Trade Execution with pauses for screenshots.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add paths
SERVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(SERVER_ROOT / ".." / ".." / "shared" / "py"))

from celery_app import celery_app
from kafka_service import KafkaPublisher, default_kafka_bus

# Demo signals (actionable BUY/SELL only)
DEMO_SIGNALS = [
    {
        "symbol": "GRASIM",
        "filing_time": "2025-12-09 21:48:46",
        "signal": 1,
        "explanation": "BlackRock subsidiary investing ₹3,000 crore - major positive catalyst",
        "confidence": 0.95,
        "subject_of_announcement": "General Updates",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/GRASIM_09122025214838_SEIntimationGIP.pdf",
        "date_time_of_submission": "2025-12-09 21:48:46"
    },
    {
        "symbol": "INDIGO",
        "filing_time": "2025-12-09 21:33:21",
        "signal": -1,
        "explanation": "DGCA orders 10% capacity reduction due to operational mismanagement - punitive action",
        "confidence": 0.95,
        "subject_of_announcement": "General Updates",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/Indigo1_09122025213248_DGCA_notice_09122025.pdf",
        "date_time_of_submission": "2025-12-09 21:33:21"
    },
    {
        "symbol": "ZAGGLE",
        "filing_time": "2025-12-09 22:39:28",
        "signal": 1,
        "explanation": "5-year strategic agreement with Mastercard - significant partnership",
        "confidence": 0.95,
        "subject_of_announcement": "Bagging/Receiving of orders/contracts",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/ZAGGLE_09122025223918_MasterCard.pdf",
        "date_time_of_submission": "2025-12-09 22:39:28"
    },
    {
        "symbol": "DHARAN",
        "filing_time": "2025-12-09 22:24:01",
        "signal": -1,
        "explanation": "Auditor warns 'going concern' uncertainty - critical negative catalyst",
        "confidence": 0.98,
        "subject_of_announcement": "Outcome of Board Meeting",
        "attachment_url": "https://nsearchares.nseindia.com/corporate/KBCGLOBAL_09122025222342_results300625.pdf",
        "date_time_of_submission": "2025-12-09 22:24:01"
    },
]

KAFKA_TOPIC = "nse_filings_trading_signal"
PAUSE_SECONDS = 2


def print_header():
    """Print demo header."""
    print("\n" + "=" * 80)
    print("🎬 NSE PIPELINE DEMO - Live Signal Processing")
    print("=" * 80)
    print(f"📊 Total Signals: {len(DEMO_SIGNALS)}")
    print(f"⏱️  Pause between signals: {PAUSE_SECONDS} seconds")
    print(f"📡 Kafka Topic: {KAFKA_TOPIC}")
    print("=" * 80 + "\n")


def print_signal_banner(idx, total, signal):
    """Print signal processing banner."""
    symbol = signal["symbol"]
    signal_value = signal["signal"]
    confidence = signal["confidence"]
    
    signal_type = "🟢 BUY" if signal_value == 1 else "🔴 SELL"
    
    print("\n" + "─" * 80)
    print(f"📊 SIGNAL {idx}/{total}")
    print("─" * 80)
    print(f"Symbol:     {symbol}")
    print(f"Action:     {signal_type}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Reason:     {signal['explanation']}")
    print("─" * 80)


def publish_to_kafka(signal):
    """Publish signal to Kafka."""
    try:
        # Get or create publisher
        from kafka_service import PublisherAlreadyRegistered
        
        try:
            publisher = default_kafka_bus.register_publisher(
                topic=KAFKA_TOPIC,
                name="demo_nse_publisher",
            )
        except PublisherAlreadyRegistered:
            publisher = default_kafka_bus.get_publisher("demo_nse_publisher")
        
        # Add timestamp
        signal_payload = signal.copy()
        signal_payload["generated_at"] = datetime.utcnow().isoformat() + "Z"
        signal_payload["source"] = "nse_filings_pipeline"
        
        publisher.publish(signal_payload, key=signal["symbol"])
        
        print(f"✅ [KAFKA] Signal published to topic '{KAFKA_TOPIC}'")
        return True
        
    except Exception as exc:
        print(f"⚠️  [KAFKA] Skipping Kafka publish: {exc}")
        # Don't fail on Kafka - continue with trade execution
        return True


def queue_trade_execution(signal):
    """Queue trade for execution."""
    try:
        task = celery_app.send_task(
            "pipeline.trade_execution.process_signal",
            args=[signal],
            queue="trading",
            priority=9,
        )
        
        print(f"✅ [CELERY] Trade queued to 'trading' queue (priority=9)")
        print(f"   Task ID: {task.id}")
        return task.id
        
    except Exception as exc:
        print(f"❌ [CELERY] Failed to queue trade: {exc}")
        return None


def wait_for_screenshot(step_num):
    """Pause and prompt for screenshot."""
    print(f"\n📸 SCREENSHOT TIME - Step {step_num}")
    print(f"   1. Check Kafka UI: http://localhost:8501 (topic: {KAFKA_TOPIC})")
    print(f"   2. Check Flower: http://localhost:5555 (trading queue)")
    print(f"   3. Check Prisma Studio: Trades/Positions tables")
    print(f"\n⏳ Waiting {PAUSE_SECONDS} seconds before next signal...")
    
    for i in range(PAUSE_SECONDS, 0, -1):
        print(f"   {i}...", end="\r")
        time.sleep(1)
    print()


def main():
    """Run demo."""
    print_header()
    
    total_signals = len(DEMO_SIGNALS)
    successful = 0
    failed = 0
    
    for idx, signal in enumerate(DEMO_SIGNALS, 1):
        print_signal_banner(idx, total_signals, signal)
        
        # Step 1: Publish to Kafka
        kafka_ok = publish_to_kafka(signal)
        
        # Step 2: Queue trade execution
        trade_ok = queue_trade_execution(signal) if kafka_ok else None
        
        if kafka_ok and trade_ok:
            successful += 1
            print(f"\n✅ Signal {idx} processed successfully")
        else:
            failed += 1
            print(f"\n❌ Signal {idx} failed")
        
        # Pause for screenshot (except last signal)
        if idx < total_signals:
            wait_for_screenshot(idx)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎬 DEMO COMPLETE")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"📊 Total:      {total_signals}")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   - Check Prisma Studio for created trades/positions")
    print("   - Monitor Celery worker logs for execution details")
    print("   - View Kafka messages in Kafka UI")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
