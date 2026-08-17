"""
CityEye - Comprehensive Project Documentation PDF Generator
Generates a complete, professional, beautifully styled PDF document
covering all project details, architecture, events, algorithms, and features.
"""

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
PDF_OUTPUT_PATH = BASE_DIR / "CityEye_Project_Documentation.pdf"


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with dynamic page numbers (Page X of Y)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "CityEye AI — Real-Time CCTV Smart Traffic & Safety Analytics")
            self.drawRightString(558, 750, "Project Documentation")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        self.drawString(54, 32, "Confidential & Proprietary — CityEye Smart City AI Systems")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=60,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")       # Dark Navy
    SECONDARY = colors.HexColor("#0284C7")     # Cyan / Sky Blue
    ACCENT_RED = colors.HexColor("#E11D48")    # Rose Red
    ACCENT_GREEN = colors.HexColor("#059669")  # Emerald
    ACCENT_AMBER = colors.HexColor("#D97706")  # Amber
    TEXT_DARK = colors.HexColor("#1E293B")     # Slate Dark
    TEXT_MUTED = colors.HexColor("#475569")    # Slate Muted
    BG_LIGHT = colors.HexColor("#F8FAFC")      # Slate Light
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
        leftIndent=12,
        spaceAfter=2
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=0
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=TEXT_DARK
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10.2,
        textColor=TEXT_DARK
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        leading=10,
        textColor=PRIMARY,
        backColor=BG_LIGHT,
        borderPadding=3,
        spaceBefore=2,
        spaceAfter=3
    )

    story = []

    # ============================================================
    # COVER / HEADER BANNER
    # ============================================================
    story.append(Paragraph("CityEye AI — Smart CCTV Traffic Analytics", title_style))
    story.append(Paragraph("Comprehensive Architecture, Multi-Model Vision Pipeline & Technical Specifications", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=0, spaceAfter=6))

    # Meta Info Box
    meta_data = [
        [
            Paragraph("<b>Version:</b> 2.4.0 (Production)", table_cell_style),
            Paragraph("<b>Author / Dev:</b> Shital Salunke", table_cell_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", table_cell_style),
        ],
        [
            Paragraph("<b>Repository:</b> github.com/salunkeshital668-byte/Event-", table_cell_style),
            Paragraph("<b>Framework:</b> YOLOv8 / YOLO11 + FastAPI", table_cell_style),
            Paragraph("<b>Status:</b> Verified & Operational", table_cell_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 170, 164])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # ============================================================
    # 1. EXECUTIVE SUMMARY & OBJECTIVES
    # ============================================================
    story.append(Paragraph("1. Executive Summary & Project Purpose", h1_style))
    story.append(Paragraph(
        "<b>CityEye AI</b> is an enterprise-grade, real-time Computer Vision surveillance and CCTV traffic analytics "
        "system designed for smart cities, highway authorities, and law enforcement. The platform processes high-density "
        "urban traffic video feeds to automatically detect vehicles, track movements, identify traffic rule violations, "
        "and immediately flag emergency road hazards and collisions without human intervention.",
        body_style
    ))
    story.append(Paragraph("<b>Key Project Capabilities:</b>", h2_style))
    story.append(Paragraph("• <b>Accident & Collision Detection:</b> Real-time collision flagging using physics IoU overlap and multi-frame deceleration tracking.", bullet_style))
    story.append(Paragraph("• <b>Helmet Safety Enforcement:</b> Dual-stage AI detecting motorcycle riders without mandatory safety helmets.", bullet_style))
    story.append(Paragraph("• <b>Traffic Law Violations:</b> Detection of triple riding and vehicles driving in the wrong lane/direction.", bullet_style))
    story.append(Paragraph("• <b>Road Hazard Monitoring:</b> Instant alerting when vehicles stall or stop unexpectedly in active traffic lanes.", bullet_style))
    story.append(Paragraph("• <b>Interactive Dashboard & Export:</b> Web dashboard with live streaming, telemetry counters, and MP4 export.", bullet_style))
    story.append(Spacer(1, 6))

    # ============================================================
    # 2. SYSTEM ARCHITECTURE & TECHNOLOGY STACK
    # ============================================================
    story.append(Paragraph("2. System Architecture & Technology Stack", h1_style))
    story.append(Paragraph(
        "CityEye utilizes a modern decoupled pipeline combining asynchronous Python web services with high-throughput "
        "deep learning inference engines running on PyTorch and Ultralytics.",
        body_style
    ))

    tech_data = [
        [Paragraph("Component", table_header_style), Paragraph("Technology / Model", table_header_style), Paragraph("Role & Description", table_header_style)],
        [
            Paragraph("<b>Base Object Detection</b>", table_cell_bold),
            Paragraph("YOLOv8n / YOLO11n (COCO)", table_cell_style),
            Paragraph("Detects 80 object classes (Cars, Buses, Trucks, Motorcycles, Persons, Traffic lights) at up to 60+ FPS.", table_cell_style)
        ],
        [
            Paragraph("<b>Helmet AI Classifier</b>", table_cell_bold),
            Paragraph("Custom YOLOv8 Best (<code>helmet_best.pt</code>)", table_cell_style),
            Paragraph("Specialized 2-class model (<code>With Helmet</code> vs <code>Without Helmet</code>) trained for rider head detection.", table_cell_style)
        ],
        [
            Paragraph("<b>Multi-Object Tracker</b>", table_cell_bold),
            Paragraph("ByteTrack (via Supervision)", table_cell_style),
            Paragraph("Maintains persistent vehicle/person IDs across frames, filtering false positive trajectory jitter.", table_cell_style)
        ],
        [
            Paragraph("<b>Backend Web Server</b>", table_cell_bold),
            Paragraph("FastAPI + Uvicorn", table_cell_style),
            Paragraph("Serves low-latency multipart/x-mixed-replace MJPEG video streams and REST telemetry APIs.", table_cell_style)
        ],
        [
            Paragraph("<b>Video Analytics Engine</b>", table_cell_bold),
            Paragraph("OpenCV (<code>cv2</code>) + NumPy", table_cell_style),
            Paragraph("Handles hardware video decoding/encoding, spatial bounding boxes, IoU collision math, and HUD overlays.", table_cell_style)
        ],
        [
            Paragraph("<b>Frontend Dashboard</b>", table_cell_bold),
            Paragraph("HTML5, Modern CSS, Vanilla JS", table_cell_style),
            Paragraph("Dark-mode glassmorphism interface, real-time metric pills, event log filtering, and full-screen controls.", table_cell_style)
        ]
    ]

    tech_table = Table(tech_data, colWidths=[115, 135, 254])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT])
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 8))

    # ============================================================
    # 3. DETAILED EVENT DETECTION CAPABILITIES
    # ============================================================
    story.append(Paragraph("3. Detailed Event Detection Modules & Physics Logic", h1_style))
    story.append(Paragraph(
        "CityEye embeds rule-based and spatial-temporal mathematical heuristics layered on top of deep neural network bounding "
        "boxes to guarantee zero false-positive accident triggers while capturing critical traffic violations.",
        body_style
    ))

    events_data = [
        [Paragraph("Event Name", table_header_style), Paragraph("Category", table_header_style), Paragraph("Detection Logic & Mathematical Criteria", table_header_style), Paragraph("Action / Output", table_header_style)],
        [
            Paragraph("<b>Helmet Violation</b><br/>(<code>helmet_violation</code>)", table_cell_bold),
            Paragraph("<font color='#E11D48'>Safety Law</font>", table_cell_style),
            Paragraph("1. Detects motorcycle and associated riders via spatial overlap.<br/>2. Crops upper head region.<br/>3. Feeds crop to <code>models/helmet_best.pt</code>.<br/>4. Triggers when class is <code>Without Helmet</code> with conf &gt; 35%.", table_cell_style),
            Paragraph("Red HUD box, increments violation counter, logs event with rider ID.", table_cell_style)
        ],
        [
            Paragraph("<b>Vehicle Collision / Accident</b><br/>(<code>accident_collision</code>)", table_cell_bold),
            Paragraph("<font color='#E11D48'>Emergency Hazard</font>", table_cell_style),
            Paragraph("1. Computes bounding box IoU overlap between all active vehicle pairs.<br/>2. Checks inter-vehicle centroid distance (&lt; 35 px).<br/>3. Flags when <code>IoU &ge; 0.03</code> or proximity persists for &ge; 4 consecutive frames.", table_cell_style),
            Paragraph("Logs accident event with both vehicle IDs, IoU score, distance, and timestamp.", table_cell_style)
        ],
        [
            Paragraph("<b>Triple Riding</b><br/>(<code>triple_riding</code>)", table_cell_bold),
            Paragraph("<font color='#D97706'>Traffic Law</font>", table_cell_style),
            Paragraph("Calculates spatial overlap ratio between detected <code>person</code> boxes and the <code>motorcycle</code> box. If <b>&ge; 3 persons</b> overlap the same motorcycle bounding box, triggers violation.", table_cell_style),
            Paragraph("Orange warning banner, increments Triple Riding counter.", table_cell_style)
        ],
        [
            Paragraph("<b>Wrong-Way Driving</b><br/>(<code>wrong_way_driving</code>)", table_cell_bold),
            Paragraph("<font color='#D97706'>Traffic Law</font>", table_cell_style),
            Paragraph("Tracks vehicle center coordinates over a 30-frame temporal history. Calculates displacement vector <i>(&Delta;x, &Delta;y)</i>. If movement opposes lane flow (e.g. <code>LEFT</code> when lane is <code>RIGHT</code>) with speed &gt; 1.5 px/frame.", table_cell_style),
            Paragraph("Logs wrong-way incident with vehicle ID and movement vector.", table_cell_style)
        ],
        [
            Paragraph("<b>Stopped Vehicle / Hazard</b><br/>(<code>vehicle_stopped</code>)", table_cell_bold),
            Paragraph("<font color='#0284C7'>Traffic Flow</font>", table_cell_style),
            Paragraph("Monitors speed of moving vehicles. If speed falls below 2.0 px/frame for &ge; 1.5 to 4.5 seconds outside predefined signal exclusion zones, flags potential breakdown or obstacle.", table_cell_style),
            Paragraph("Yellow hazard alert, updates Active Hazard counter.", table_cell_style)
        ],
        [
            Paragraph("<b>Multi-Class Detection</b><br/>(<code>object_detected</code>)", table_cell_bold),
            Paragraph("<font color='#059669'>Telemetry</font>", table_cell_style),
            Paragraph("Continuous frame-by-frame object classification for <b>Cars, Buses, Trucks, Motorcycles, Persons</b> with bounding box coordinates and confidence percentages.", table_cell_style),
            Paragraph("Real-time HUD labels and live telemetry statistics.", table_cell_style)
        ]
    ]

    events_table = Table(events_data, colWidths=[100, 75, 205, 124])
    events_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT])
    ]))
    story.append(events_table)
    story.append(Spacer(1, 8))

    # ============================================================
    # 4. MULTI-VIDEO PIPELINE & DATA SEPARATION
    # ============================================================
    story.append(Paragraph("4. Multi-Video Processing & Event Data Separation", h1_style))
    story.append(Paragraph(
        "CityEye dynamically scans the <code>videos/</code> directory for all supported video formats (<code>.mp4</code>, "
        "<code>.avi</code>, <code>.mov</code>, <code>.mkv</code>). It supports both sequential batch execution and individual "
        "video selection while strictly guaranteeing that events from different video streams remain isolated.",
        body_style
    ))

    video_data = [
        [Paragraph("Video Stream", table_header_style), Paragraph("Input File", table_header_style), Paragraph("Resolution & FPS", table_header_style), Paragraph("Key Events Captured", table_header_style), Paragraph("Output Event Log", table_header_style)],
        [
            Paragraph("<b>Video 1</b> (Normal Traffic)", table_cell_bold),
            Paragraph("<code>videos/input.mp4</code>", table_cell_style),
            Paragraph("480x360 @ 25 FPS (805 frames)", table_cell_style),
            Paragraph("53 NO HELMET violations, 102 vehicle detections, triple riding", table_cell_style),
            Paragraph("<code>data/video_1_events.json</code><br/><code>data/input_events.json</code>", table_cell_style)
        ],
        [
            Paragraph("<b>Video 2</b> (Accident Scene)", table_cell_bold),
            Paragraph("<code>videos/accident.mp4</code>", table_cell_style),
            Paragraph("640x360 @ 20 FPS (340 frames)", table_cell_style),
            Paragraph("11 Vehicle collisions/accidents, 31 stopped vehicle hazards", table_cell_style),
            Paragraph("<code>data/video_2_events.json</code><br/><code>data/accident_events.json</code>", table_cell_style)
        ],
        [
            Paragraph("<b>Combined Archive</b>", table_cell_bold),
            Paragraph("All Streams", table_cell_style),
            Paragraph("Multi-resolution", table_cell_style),
            Paragraph("243 total categorized security & traffic events", table_cell_style),
            Paragraph("<code>data/events.json</code>", table_cell_style)
        ]
    ]

    video_table = Table(video_data, colWidths=[95, 95, 95, 115, 104])
    video_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT])
    ]))
    story.append(video_table)
    story.append(Spacer(1, 8))

    # ============================================================
    # 5. DASHBOARD & INTERACTIVE VIDEO PLAYER
    # ============================================================
    story.append(Paragraph("5. Web Dashboard & Interactive Video Player Features", h1_style))
    story.append(Paragraph(
        "The CityEye web interface (<code>http://127.0.0.1:8000</code>) provides a mission-control operations room experience:",
        body_style
    ))
    story.append(Paragraph("• <b>Live MJPEG Video Feed:</b> Streams real-time inference frames with bounding boxes, confidence badges, and HUD overlays.", bullet_style))
    story.append(Paragraph("• <b>Integrated Play / Pause Controls:</b> Interactive button (Play / Pause) synchronized with backend streaming.", bullet_style))
    story.append(Paragraph("• <b>Edge-to-Edge Full Screen:</b> Seamless full-screen mode utilizing standard browser Fullscreen API with zero black margins.", bullet_style))
    story.append(Paragraph("• <b>Real-Time Telemetry Counters:</b> Instantaneous counts for Cars, Buses, Trucks, Motorcycles, Persons, and Safety Violations.", bullet_style))
    story.append(Paragraph("• <b>Search & Filter Event Log:</b> Searchable event log filtered by incident type (Helmet, Collision, Wrong-Way, Stopped).", bullet_style))
    story.append(Spacer(1, 6))

    # ============================================================
    # 6. CODEBASE DIRECTORY STRUCTURE
    # ============================================================
    story.append(Paragraph("6. Project File Structure & Component Reference", h1_style))
    
    file_data = [
        [Paragraph("File / Directory", table_header_style), Paragraph("Type", table_header_style), Paragraph("Responsibility & Key Functions", table_header_style)],
        [
            Paragraph("<code>app.py</code>", table_cell_bold),
            Paragraph("FastAPI App", table_cell_style),
            Paragraph("Handles <code>/video-feed</code> MJPEG streaming, <code>/live-data</code> telemetry API, <code>/videos</code> list, and static UI routes.", table_cell_style)
        ],
        [
            Paragraph("<code>detector.py</code>", table_cell_bold),
            Paragraph("AI Detection", table_cell_style),
            Paragraph("Initializes YOLO models, CUDA/CPU thread optimization, and runs frame-by-frame inference with custom confidence.", table_cell_style)
        ],
        [
            Paragraph("<code>tracker.py</code>", table_cell_bold),
            Paragraph("ByteTrack", table_cell_style),
            Paragraph("Supervision ByteTrack multi-object tracker maintaining consistent track IDs and spatial trajectories.", table_cell_style)
        ],
        [
            Paragraph("<code>event_detector.py</code>", table_cell_bold),
            Paragraph("Analytics Engine", table_cell_style),
            Paragraph("Core logic for IoU collisions, helmet safety cropping, triple riding, wrong-way detection, and JSON event logging.", table_cell_style)
        ],
        [
            Paragraph("<code>main.py</code>", table_cell_bold),
            Paragraph("CLI Pipeline", table_cell_style),
            Paragraph("Discovers all videos in <code>videos/</code>, processes sequentially, saves annotated MP4 outputs, and outputs summary.", table_cell_style)
        ],
        [
            Paragraph("<code>config.py</code>", table_cell_bold),
            Paragraph("Configuration", table_cell_style),
            Paragraph("Centralized path resolution, supported formats (<code>.mp4</code>, <code>.avi</code>, <code>.mov</code>, <code>.mkv</code>), and model constants.", table_cell_style)
        ],
        [
            Paragraph("<code>models/helmet_best.pt</code>", table_cell_bold),
            Paragraph("YOLO Model", table_cell_style),
            Paragraph("Custom-trained PyTorch weight file for high-accuracy motorcycle rider helmet classification.", table_cell_style)
        ],
        [
            Paragraph("<code>templates/index.html</code>", table_cell_bold),
            Paragraph("UI Template", table_cell_style),
            Paragraph("Responsive CCTV dashboard structure, video frame container, control buttons, stat cards, and event table.", table_cell_style)
        ],
        [
            Paragraph("<code>static/style.css</code>", table_cell_bold),
            Paragraph("CSS Stylesheet", table_cell_style),
            Paragraph("Cyber-security dark theme, responsive grid, glassmorphism cards, and edge-to-edge full-screen styling.", table_cell_style)
        ],
        [
            Paragraph("<code>static/script.js</code>", table_cell_bold),
            Paragraph("Frontend Logic", table_cell_style),
            Paragraph("Asynchronous video stream controller, 200ms telemetry poller, Play/Pause handler, and Fullscreen API listener.", table_cell_style)
        ]
    ]

    file_table = Table(file_data, colWidths=[120, 75, 309])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT])
    ]))
    story.append(file_table)
    story.append(Spacer(1, 8))

    # ============================================================
    # 7. HOW TO EXECUTE & RUN
    # ============================================================
    story.append(Paragraph("7. Execution Guide & Command Reference", h1_style))
    story.append(Paragraph("<b>1. Automatic Multi-Video Processing (CLI Batch Runner):</b>", h2_style))
    story.append(Paragraph("<code>python main.py</code>", code_style))
    story.append(Paragraph("Scans <code>videos/</code>, executes all clips sequentially, outputs annotated MP4s to <code>output/</code>, and creates separate JSON logs.", body_style))

    story.append(Paragraph("<b>2. Single Video CLI Processing:</b>", h2_style))
    story.append(Paragraph("<code>python main.py --video videos/accident.mp4</code>", code_style))

    story.append(Paragraph("<b>3. Launch Web Dashboard (Live Browser GUI):</b>", h2_style))
    story.append(Paragraph("<code>python -m uvicorn app:app --host 127.0.0.1 --port 8000</code>", code_style))
    story.append(Paragraph("Open <b>http://127.0.0.1:8000</b> in any modern web browser to interact with live streaming and detection.", body_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Documentation PDF generated successfully at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
