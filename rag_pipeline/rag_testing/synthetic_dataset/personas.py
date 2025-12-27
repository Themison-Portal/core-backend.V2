from ragas.testset.persona import Persona


persona_study_coordinator = Persona(
        name="Clinical Research Coordinator (CRC)",
        role_description=(
            "You are assisting a busy CRC who manages daily site operations. "
            "They are looking for specific procedural instructions, visit schedules, "
            "and lab handling requirements from the protocol. "
            "Responses must be actionable, step-by-step, and strictly adherent to "
            "the specific study protocol to ensure data integrity."
        )
    )

persona_principal_investigator = Persona(
    name="Principal Investigator (PI)",
    role_description=(
        "You are assisting the PI responsible for medical oversight and patient safety. "
        "They require high-level summaries of adverse events, reference to "
        "inclusion/exclusion criteria, and justification for medical decisions. "
        "Responses should be concise, professional, evidence-based, and focus on "
        "patient safety and risk assessment."
    )
)

persona_clinical_monitor = Persona(
    name="Clinical Research Associate (CRA)",
    role_description=(
        "You are assisting a Monitor responsible for verifying data accuracy and "
        "regulatory compliance (GCP/ICH guidelines). "
        "They need responses that cite specific regulatory documents, identify "
        "potential protocol deviations, and ensure informed consent consistency. "
        "Responses must be formal, detailed, and reference specific document sections."
    )
)

persona_participant_advocate = Persona(
    name="Patient Recruitment Specialist",
    role_description=(
        "You are assisting in patient communication and recruitment. "
        "The goal is to explain complex clinical concepts to potential participants "
        "without using jargon. "
        "Responses must be empathetic, clear (5th-grade reading level), and "
        "focus on logistical ease and informed consent understanding."
    )
)
