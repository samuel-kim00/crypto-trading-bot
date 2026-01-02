import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF

class WeeklyReportPDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.darkgreen,
            spaceBefore=20,
            spaceAfter=10
        )
        
        # Subheader style
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.darkred,
            spaceBefore=15,
            spaceAfter=8
        )
        
        # Body style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6
        )
        
        # Small text style
        self.small_style = ParagraphStyle(
            'CustomSmall',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            spaceBefore=3,
            spaceAfter=3
        )
    
    def generate_pdf(self, report_data, output_path):
        """Generate PDF from report data"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        title = Paragraph("Weekly AI Trading Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report date
        report_date = datetime.fromisoformat(report_data['report_date']).strftime("%B %d, %Y at %I:%M %p")
        date_para = Paragraph(f"Generated on: {report_date}", self.body_style)
        story.append(date_para)
        story.append(Spacer(1, 20))
        
        # Market Overview
        story.append(Paragraph("Market Overview & Analysis", self.header_style))
        
        market_data = [
            ['Metric', 'Value'],
            ['Fear & Greed Index', f"{report_data['market_overview']['fear_greed_index']}/100"],
            ['Bitcoin Dominance', f"{report_data['market_overview']['bitcoin_dominance']:.1f}%"],
            ['Market Sentiment', report_data['market_overview']['market_sentiment']],
            ['Total Coins Analyzed', str(report_data['market_overview']['total_analyzed'])],
            ['Day Trading Opportunities', str(report_data['market_overview']['day_trading_opportunities'])],
            ['Long-term Opportunities', str(report_data['market_overview']['long_term_opportunities'])]
        ]
        
        market_table = Table(market_data, colWidths=[3*inch, 2*inch])
        market_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(market_table)
        story.append(Spacer(1, 20))
        
        # Feedback Analysis (if available)
        feedback_analysis = report_data.get('feedback_analysis', {})
        if 'total_predictions' in feedback_analysis and feedback_analysis['total_predictions'] > 0:
            story.append(Paragraph("Past Performance Analysis", self.header_style))
            
            feedback_text = f"Total Past Predictions: {feedback_analysis['total_predictions']}<br/>"
            
            if 'accuracy_by_timeframe' in feedback_analysis:
                for timeframe, data in feedback_analysis['accuracy_by_timeframe'].items():
                    accuracy = data['average_accuracy']
                    feedback_text += f"{timeframe}: {accuracy:.1f}% accuracy<br/>"
            
            if 'lessons_learned' in feedback_analysis:
                feedback_text += "<br/>Key Insights:<br/>"
                for lesson in feedback_analysis['lessons_learned']:
                    feedback_text += f"• {lesson}<br/>"
            
            story.append(Paragraph(feedback_text, self.body_style))
            story.append(Spacer(1, 20))
        elif 'message' in feedback_analysis:
            story.append(Paragraph("Feedback Analysis", self.header_style))
            story.append(Paragraph(feedback_analysis['message'], self.body_style))
            story.append(Spacer(1, 20))
        
        # Day Trading Opportunities
        story.append(Paragraph("⚡ Day Trading Opportunities (1-3 days)", self.header_style))
        
        if report_data['day_trading']['buy_recommendations']:
            for i, rec in enumerate(report_data['day_trading']['buy_recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec['symbol']} - {rec['signals']['confidence']:.0f}% Confidence", self.subheader_style))
                
                # Create recommendation table
                rec_data = [
                    ['Entry Price', f"${rec['signals']['entry_price']:.4f}"],
                    ['Stop Loss', f"${rec['signals']['stop_loss']:.4f}"],
                    ['Take Profit Targets', ', '.join([f"${tp:.4f}" for tp in rec['signals']['take_profit']])],
                    ['1-Day Prediction', f"{rec['predictions']['target_1d']*100:.1f}%"],
                    ['Risk Level', rec['signals']['risk_level']],
                    ['Timeframe', rec['signals']['timeframe']]
                ]
                
                rec_table = Table(rec_data, colWidths=[2*inch, 3*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                
                story.append(rec_table)
                
                # Reasoning
                reasoning_text = "Key Signals: " + ', '.join(rec['signals']['reasoning'][:3])
                story.append(Paragraph(reasoning_text, self.small_style))
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph("No day trading opportunities found with sufficient confidence.", self.body_style))
            story.append(Spacer(1, 20))
        
        # Page break before long-term section
        story.append(PageBreak())
        
        # Long-term Opportunities
        story.append(Paragraph("📈 Long-term Opportunities (1-4 weeks)", self.header_style))
        
        if report_data['long_term']['buy_recommendations']:
            for i, rec in enumerate(report_data['long_term']['buy_recommendations'], 1):
                story.append(Paragraph(f"{i}. {rec['symbol']} - {rec['signals']['confidence']:.0f}% Confidence", self.subheader_style))
                
                # Create recommendation table
                rec_data = [
                    ['Entry Price', f"${rec['signals']['entry_price']:.4f}"],
                    ['Stop Loss', f"${rec['signals']['stop_loss']:.4f}"],
                    ['Take Profit Targets', ', '.join([f"${tp:.4f}" for tp in rec['signals']['take_profit']])],
                    ['7-Day Prediction', f"{rec['predictions']['target_7d']*100:.1f}%"],
                    ['Risk Level', rec['signals']['risk_level']],
                    ['Timeframe', rec['signals']['timeframe']]
                ]
                
                rec_table = Table(rec_data, colWidths=[2*inch, 3*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                
                story.append(rec_table)
                
                # Reasoning
                reasoning_text = "Key Signals: " + ', '.join(rec['signals']['reasoning'][:3])
                story.append(Paragraph(reasoning_text, self.small_style))
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph("No long-term opportunities found with sufficient confidence.", self.body_style))
            story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        disclaimer = """
        <b>DISCLAIMER:</b> This report is generated by AI for educational and informational purposes only. 
        It is not financial advice. Cryptocurrency trading involves significant risk of loss. 
        Past performance does not guarantee future results. Always do your own research and consult 
        with financial professionals before making investment decisions.
        """
        story.append(Paragraph(disclaimer, self.small_style))
        
        # Build PDF
        doc.build(story)
        return output_path

def generate_weekly_report_pdf(report_data, filename=None):
    """Convenience function to generate PDF report"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weekly_report_{timestamp}.pdf"
    
    # Use current working directory for reports
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, filename)
    
    generator = WeeklyReportPDFGenerator()
    return generator.generate_pdf(report_data, output_path) 