import schedule
import time
import asyncio
import logging
from datetime import datetime
from weekly_predictor import WeeklyPredictor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

class WeeklyScheduler:
    def __init__(self):
        self.predictor = WeeklyPredictor()
        self.logger = logging.getLogger(__name__)
        
        # Email configuration (optional)
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', '')
    
    async def generate_and_send_report(self):
        """Generate weekly report and optionally send via email"""
        try:
            self.logger.info("Starting weekly report generation...")
            
            # Generate the report
            report = await self.predictor.generate_weekly_report()
            
            # Create human-readable summary
            summary = self.create_email_summary(report)
            
            # Print to console
            print(summary)
            
            # Send email if configured
            if self.email_enabled and self.recipient_email:
                self.send_email_report(summary, report)
                
            self.logger.info("Weekly report completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {str(e)}")
    
    def create_email_summary(self, report):
        """Create a human-readable email summary"""
        summary = f"""
🚀 WEEKLY CRYPTO TRADING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 MARKET OVERVIEW
• Fear & Greed Index: {report['market_overview']['fear_greed_index']}/100
• Bitcoin Dominance: {report['market_overview']['bitcoin_dominance']:.1f}%
• Market Sentiment: {report['market_overview']['market_sentiment']}

🟢 TOP BUY RECOMMENDATIONS
"""
        
        for i, rec in enumerate(report['top_buy_recommendations'][:3], 1):
            summary += f"""
{i}. {rec['symbol']} - Confidence: {rec['signals']['confidence']:.0f}%
   Current Price: ${rec['current_price']:.4f}
   Entry Price: ${rec['signals']['entry_price']:.4f}
   Stop Loss: ${rec['signals']['stop_loss']:.4f}
   Take Profit Levels: {', '.join([f'${tp:.4f}' for tp in rec['signals']['take_profit']])}
   
   Technical Analysis:
   • RSI: {rec['technical_analysis']['rsi']:.1f}
   • MACD: {rec['technical_analysis']['macd']:.4f}
   • Trend: {rec['technical_analysis']['trend'].upper()}
   • Volume Ratio: {rec['technical_analysis']['volume_ratio']:.2f}x
   
   ML Predictions:
   • 1-day: {rec['predictions'].get('target_1d', 0)*100:.1f}%
   • 3-day: {rec['predictions'].get('target_3d', 0)*100:.1f}%
   • 7-day: {rec['predictions'].get('target_7d', 0)*100:.1f}%
   
   Reasoning: {', '.join(rec['signals']['reasoning'])}
"""
        
        if report['top_sell_recommendations']:
            summary += "\n🔴 TOP SELL RECOMMENDATIONS\n"
            for i, rec in enumerate(report['top_sell_recommendations'][:2], 1):
                summary += f"""
{i}. {rec['symbol']} - Confidence: {rec['signals']['confidence']:.0f}%
   Current Price: ${rec['current_price']:.4f}
   Reasoning: {', '.join(rec['signals']['reasoning'])}
"""
        
        summary += f"\n📋 WATCHLIST (HOLD)\n"
        for i, rec in enumerate(report['watchlist'][:5], 1):
            summary += f"{i}. {rec['symbol']} - Price: ${rec['current_price']:.4f} - Trend: {rec['technical_analysis']['trend']}\n"
        
        summary += f"\n📈 EXECUTIVE SUMMARY\n"
        for point in report['summary']:
            summary += f"• {point}\n"
        
        return summary
    
    def send_email_report(self, summary, report):
        """Send email report"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Weekly Crypto Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            msg.attach(MIMEText(summary, 'plain'))
            
            # Attach JSON report
            json_report = MIMEText(str(report))
            json_report.add_header('Content-Disposition', 'attachment', filename='weekly_report.json')
            msg.attach(json_report)
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, self.recipient_email, text)
            server.quit()
            
            self.logger.info(f"Email report sent to {self.recipient_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
    
    def run_weekly_job(self):
        """Wrapper function for schedule"""
        asyncio.run(self.generate_and_send_report())
    
    def start_scheduler(self):
        """Start the weekly scheduler"""
        # Schedule weekly report every Sunday at 9:00 AM
        schedule.every().sunday.at("09:00").do(self.run_weekly_job)
        
        # For testing - you can also schedule daily or specific times
        # schedule.every().day.at("10:00").do(self.run_weekly_job)
        # schedule.every(10).minutes.do(self.run_weekly_job)  # For testing
        
        self.logger.info("Weekly scheduler started. Next report: Sunday 9:00 AM")
        
        # Run immediately for testing
        self.logger.info("Running initial report...")
        self.run_weekly_job()
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    scheduler = WeeklyScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main() 