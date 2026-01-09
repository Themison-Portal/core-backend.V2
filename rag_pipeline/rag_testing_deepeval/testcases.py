test_data = [
    {
        "name": "Q1: Trial Identification",
        "input": "What is the Trial ID and EudraCT number for this clinical trial?",
        "expected_output": "Trial ID: Sym004-09; EudraCT number: 2015-003047-19",
        "pages": [1],
        "sections": [] 
    },
    {
        "name": "Q2: Drug Identity",
        "input": "What is Sym004 and what are the two monoclonal antibodies it contains?",
        "expected_output": "Sym004 is a 1:1 mixture of 2 monoclonal antibodies (mAbs): mAb992 (futuximab) and mAb1024 (modotuximab), which bind specifically to 2 non-overlapping epitopes of the EGFR.",
        "pages": [19, 20],
        "sections": ["3.2.1"]
    },
    {
        "name": "Q3: Patient Population",
        "input": "What is the total planned number of patients for this trial?",
        "expected_output": "50 patients total (up to 18 in dose-escalation, 32-41 in dose-expansion)",
        "pages": [15, 16, 37],
        "sections": ["5.5"]
    },
    {
        "name": "Q4: Dosing Schedule",
        "input": "How often is Sym004 administered and on which days of the cycle?",
        "expected_output": "Sym004 is administered every second week (Day 1 and Day 15 of each 28-day cycle, ±2 days) by IV infusion.",
        "pages": [32, 45],
        "sections": ["7.1.4.1"]
    },
    {
        "name": "Q5: Starting Dose",
        "input": "What is Dose Level 1 (the starting dose) for Sym004 in the dose-escalation phase?",
        "expected_output": "Dose Level 1: Sym004 12 mg/kg + FOLFIRI",
        "pages": [45],
        "sections": ["7.1.4.2"]
    },
    {
        "name": "Q6: Performance Status",
        "input": "What ECOG performance status is required for patient inclusion?",
        "expected_output": "ECOG performance status of 0 or 1",
        "pages": [38],
        "sections": ["6.1"]
    },
    {
        "name": "Q7: Sponsor Information",
        "input": "Who is the sponsor of this clinical trial?",
        "expected_output": "Symphogen A/S",
        "pages": [1],
        "sections": []
    }
]