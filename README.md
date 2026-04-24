VisionDrive AI is an end-to-end Smart City solution designed to optimize urban traffic flow and enhance road safety. By leveraging Computer Vision at the edge, the system transforms standard CCTV feeds into actionable data for traffic signal control and emergency services.


🌟 Key Features

    📊 Adaptive Traffic Control: Real-time vehicle density estimation using YOLOv8 to dynamically adjust signal timings, reducing intersection congestion.

    ⛑️ Automated Violation Detection: Nested detection pipeline to identify two-wheeler riders without helmets and trigger ANPR (License Plate Recognition).

    💥 Real-time Accident Response: Heuristic-based collision detection that automatically sends SMS alerts with location coordinates via Twilio API.

    🖥️ Smart City Dashboard: A clean, centralized interface for traffic handlers to monitor live feeds and violation logs.

    ⚡ Edge-Optimized: Built with multithreading to run efficiently on hardware like NVIDIA Jetson or high-performance laptops.
    

🛠️ Tech Stack

    AI/ML: Python, YOLOv8, OpenCV, EasyOCR

    IoT & Backend: Twilio API (SMS), Threading

    Hardware Target: Edge AI (Raspberry Pi / Jetson Nano / PC)

    Tools: GitHub CLI, PyTorch
