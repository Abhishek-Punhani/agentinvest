#!/usr/bin/env python3
"""
Manually push selected NSE signals to trading queue for execution.
"""

import sys
from pathlib import Path

# Add paths for imports
SERVER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(SERVER_ROOT / ".." / ".." / "shared" / "py"))

from celery_app import celery_app

# Selected signals to execute (after RELINFRA)
SIGNALS = [
    {
        "symbol": "GRASIM",
        "filing_time": "2025-12-09 21:48:46",
        "signal": 1,
        "explanation": "The external search confirms the factual basis of the explanation. Multiple sources verify that Global Infrastructure Partners (part of BlackRock) has agreed to invest up to ₹3,000 crore in Aditya Birla Renewables, a subsidiary of Grasim. This is a high-impact, positive catalyst involving a major global investor, which validates the business and provides significant growth capital. The logic that this infusion will fuel expansion to over 10 GW is explicitly supported by the news. The signal is strongly aligned with the event, and the original model's high confidence is justified and slightly increased due to direct verification of the claims.",
        "confidence": 0.95,
        "subject_of_announcement": "General Updates",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/GRASIM_09122025214838_SEIntimationGIP.pdf",
        "date_time_of_submission": "2025-12-09 21:48:46"
    },
    {
        "symbol": "INDIGO",
        "filing_time": "2025-12-09 21:33:21",
        "signal": -1,
        "explanation": "The external search confirms the core claim. The Indian government and the DGCA have directed InterGlobe Aviation (IndiGo) to reduce its winter schedule by 10% due to operational mismanagement. This is a significant, punitive action that will materially impact the airline's capacity and revenue. The event is a high-impact, negative catalyst. The logic is sound, the signal is appropriate, and the original confidence is justified, if not slightly understated.",
        "confidence": 0.95,
        "subject_of_announcement": "General Updates",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/Indigo1_09122025213248_DGCA_notice_09122025.pdf",
        "date_time_of_submission": "2025-12-09 21:33:21"
    },
    {
        "symbol": "ZAGGLE",
        "filing_time": "2025-12-09 22:39:28",
        "signal": 1,
        "explanation": "The explanation is logically coherent and factually verified. External news confirms the 5-year strategic agreement with Mastercard, which is a significant positive catalyst. The claim that the news is recent and its impact is likely not yet priced into the stock strengthens the bullish argument. The high confidence is justified.",
        "confidence": 0.95,
        "subject_of_announcement": "Bagging/Receiving of orders/contracts",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/ZAGGLE_09122025223918_MasterCard.pdf",
        "date_time_of_submission": "2025-12-09 22:39:28"
    },
    {
        "symbol": "DHARAN",
        "filing_time": "2025-12-09 22:24:01",
        "signal": -1,
        "explanation": "The validation confirms the strong sell signal. The reasoning is exceptionally strong and logically coherent. An auditor's 'Modified Opinion' highlighting a 'material uncertainty about the Company's ability to continue as a going concern' is a critical, high-impact negative catalyst. The detailed list of severe financial and regulatory issues provides a powerful fundamental basis for a high-conviction sell signal.",
        "confidence": 0.98,
        "subject_of_announcement": "Outcome of Board Meeting",
        "attachment_url": "https://nsearchives.nseindia.com/corporate/KBCGLOBAL_09122025222342_results300625.pdf",
        "date_time_of_submission": "2025-12-09 22:24:01"
    },
]


def push_signals():
    """Push selected signals to trading queue."""
    print(f"📊 Pushing {len(SIGNALS)} actionable signals to trading queue...\n")
    
    successful = 0
    failed = 0
    
    for idx, signal in enumerate(SIGNALS, 1):
        symbol = signal["symbol"]
        signal_value = signal["signal"]
        confidence = signal["confidence"]
        
        signal_type = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "HOLD"
        
        try:
            # Queue trade execution task with HIGH priority
            task = celery_app.send_task(
                "pipeline.trade_execution.process_signal",
                args=[signal],
                queue="trading",
                priority=9,  # HIGH PRIORITY
            )
            
            print(f"{idx}. ✅ {symbol} ({signal_type}) - Confidence: {confidence:.0%}")
            print(f"   Task ID: {task.id}")
            print(f"   Queue: trading (priority=9)")
            print()
            
            successful += 1
            
        except Exception as exc:
            print(f"{idx}. ❌ {symbol} - FAILED: {exc}\n")
            failed += 1
    
    print("=" * 60)
    print(f"📊 SUMMARY: {successful} queued, {failed} failed")
    print("=" * 60)
    print("\n💡 TIP: Monitor execution with:")
    print("   - Trading worker logs")
    print("   - Flower dashboard: http://localhost:5555")
    print("   - Redis queue: redis-cli LLEN trading")


if __name__ == "__main__":
    push_signals()
